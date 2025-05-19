"""
Tailored application for labelling the ELSA dataset.

This raw-data for this data-set has the following structure:
- labels/
    - elsa_blip2_annotations.pkl
    - /<participant label folders to be created>
- raw-axivity-accFiles/
    - participant_10<2 digit number>_axivity_data.CWA.gz
    - ...
- raw-camera-images/
    - camera_participant_10<2 digit number>/
        - B..._2..._20211201_120000E.JPG
        - ...

This application will create folders for each participant with data in the raw-camera-images directory and raw-axivity-accFiles directory
where we can store the labels saved by the actilabelv2 application.
It will also read in the elsa_blip2_annotations.pkl file to initialize the outdoor/indoor labels.
This file has the following structure:
- id: participant id
- time: timestamp of the image
- blip2_label: 'outdoor' or 'indoor'

So when loading in the data for an unlabeled participant, we need to initialse the labels for the outdoor/indoor channel by
reading in the elsa_blip2_annotations.pkl file, selecting the participant id, and getting the label-time stamps.
To convert the image timestamps to label ranges, use the spread_label_time function.

Ultimately, I want a CLI which displays the list of participants which have/have not been labelled yet, and prompts the user which participant they would like to label.
If it is a previously labelled participant, it should reload the existing labels.
If it is a new participant, it should set up the intial labels based on the elsa_blip2_annotations.pkl file.
    
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import glob
import pickle
import actipy
from actilabelv2.main import AnnotationTool, ImageDataSource, AnnotationChannel, Annotation, VectorDataSource, ScalarDataSource


def main():
    """Main CLI entry point."""
    # Get base directory from environment or use default
    base_dir = os.getenv("ELSA_BASE_DIR", ".")
    
    # Get lists of labeled and unlabeled participants
    labeled_participants, unlabeled_participants = get_participant_list(base_dir)
    
    # Print participant lists in columns
    def print_participant_columns(participants, title):
        print(f"\n{title}:")
        if not participants:
            print("  None")
            return
            
        # Sort participants
        participants = sorted(participants)
        
        # Calculate number of columns (4 columns with 20 char width each)
        col_width = 20
        num_cols = 4
        
        # Print in columns
        for i in range(0, len(participants), num_cols):
            row = participants[i:i + num_cols]
            print("  " + "".join(f"{p:<{col_width}}" for p in row))
    
    print_participant_columns(labeled_participants, "Labeled participants")
    print_participant_columns(unlabeled_participants, "Unlabeled participants")
    
    # Get participant selection
    while True:
        participant_id = input("\nEnter participant ID to label (or 'q' to quit): ").strip()
        if participant_id.lower() == 'q':
            break
            
        # Check if the entered ID exists in either list
        if participant_id not in labeled_participants and participant_id not in unlabeled_participants:
            # Try with '10' prefix if not present
            if not participant_id.startswith('10'):
                prefixed_id = '10' + participant_id
                if prefixed_id in labeled_participants or prefixed_id in unlabeled_participants:
                    participant_id = prefixed_id
                else:
                    print(f"Invalid participant ID: {participant_id}")
                    continue
            else:
                print(f"Invalid participant ID: {participant_id}")
                continue
        
        # Get image paths
        image_paths = get_participant_images(base_dir, participant_id)
        if not image_paths:
            print(f"No images found for participant {participant_id}")
            continue
            
        # Get accelerometer data
        acc_path = get_participant_acc_data(base_dir, participant_id)
        
        # Create annotation tool
        tool = AnnotationTool()
        
        # Set up image data source
        image_times = [np.datetime64(datetime.strptime(os.path.basename(p), "%Y%m%d_%H%M%S.JPG")) 
                      for p in image_paths]
        
        # Create data sources list with both preview and thumbnail image sources
        data_sources = [
            ImageDataSource(
                "Preview",
                np.array(image_times),
                image_paths,
                thumbnail_size=(300, 300),
                max_images_in_view=1,
            ),
            ImageDataSource(
                "Thumbnails",
                np.array(image_times),
                image_paths,
                thumbnail_size=(100, 100),
                max_images_in_view=10,
                display_mode='centered',
            )
        ]
        
        # Load and process CWA data if available
        if acc_path:
            try:
                print(f"Loading sensor data from {acc_path}")
                data, info = actipy.read_device(acc_path,
                                              lowpass_hz=20,
                                              calibrate_gravity=True,
                                              detect_nonwear=True,
                                              resample_hz=50)
                
                # Convert index to numpy datetime64
                sensor_times = data.index.values.astype(np.datetime64)
                
                # Add sensor data sources
                data_sources.extend([
                    VectorDataSource(
                        "Accelerometer",
                        sensor_times,
                        data[['x', 'y', 'z']].values,
                        dim_names=['x', 'y', 'z']
                    ),
                    ScalarDataSource(
                        "Temperature",
                        sensor_times,
                        data['temperature'].values,
                        color=(255, 100, 100)  # Red
                    ),
                    ScalarDataSource(
                        "Light",
                        sensor_times,
                        data['light'].values,
                        color=(255, 255, 100)  # Yellow
                    )
                ])
                print("Successfully loaded sensor data")
            except Exception as e:
                print(f"Error loading sensor data: {e}")
        else:
            print(f"No accelerometer data found for participant {participant_id}")
        
        # Set up annotation channel
        annotation_channel = AnnotationChannel(
            name="Indoor/Outdoor",
            possible_labels=["indoor", "outdoor"]
        )
        
        # Load or create labels
        if participant_id in labeled_participants:
            print(f"Loading existing labels for participant {participant_id}")
            annotations = load_existing_labels(base_dir, participant_id)
            for ann in annotations:
                annotation_channel.add_annotation(ann)
        else:
            print(f"Creating initial labels for participant {participant_id}")
            annotations = create_initial_labels(base_dir, participant_id)
            for ann in annotations:
                annotation_channel.add_annotation(ann)
        
        # Load data into tool
        tool.load_data(
            data_sources=data_sources,
            annotation_channels=[annotation_channel],
            load_existing=True
        )
        
        # Run the tool
        tool.run()


def get_participant_list(base_dir: str) -> Tuple[List[str], List[str]]:
    """Get lists of labeled and unlabeled participants.
    
    Args:
        base_dir: Base directory containing the dataset
        
    Returns:
        Tuple of (labeled_participants, unlabeled_participants)
    """
    # Get all participants with camera images
    camera_dir = os.path.join(base_dir, "raw-camera-images")
    all_participants = [d.split("camera_participant_")[1] 
                       for d in os.listdir(camera_dir) 
                       if d.startswith("camera_participant_")]
    
    # Get labeled participants
    labels_dir = os.path.join(base_dir, "labels")
    labeled_participants = [d for d in os.listdir(labels_dir) 
                          if os.path.isdir(os.path.join(labels_dir, d))]
    
    # Get unlabeled participants
    unlabeled_participants = [p for p in all_participants if p not in labeled_participants]
    
    return labeled_participants, unlabeled_participants

def get_base_participant_id(participant_id: str) -> str:
    """Convert a participant ID to its base form by removing the '10' prefix if present.
    
    Args:
        participant_id: Participant ID (e.g. '1001' or '1')
        
    Returns:
        Base participant ID (e.g. '1' for both '1001' and '1')
    """
    if participant_id.startswith('10'):
        return participant_id[2:]
    return participant_id

def load_blip2_annotations(base_dir: str) -> pd.DataFrame:
    """Load the blip2 annotations from pickle file.
    
    Args:
        base_dir: Base directory containing the dataset
        
    Returns:
        DataFrame containing blip2 annotations
    """
    pkl_path = os.path.join(base_dir, "labels", "elsa_blip2_annotations.pkl")
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    
def spread_label_time(
    labels: np.ndarray,
    times: np.ndarray,
    max_left_secs: int = 0,
    max_right_secs: int = 0,
):
    """
    Given an np.ndarray of str/int labels and datetime64 times, create label segments with start_times and end_times.

    If two consecutive labels are the same and their boundaries overlap, the labels are merged into one segment.

    Otherwise, for distinct neighbouring labels, the start and end times are calculated as follows:
    For time point i,
    - start_time[i] = max(times[i] - max_left_secs, (times[i-1] + max_right_secs + times[i] - max_left_secs)/2)
    - stop_time[i] = min(times[i] + max_right_secs, (times[i] + max_right_secs + times[i+1] - max_left_secs)/2)
    if i == 0, start_time[i] = times[i] - max_left_secs,
    if i == len(times)-1, stop_time[i] = times[i] + max_right_secs.
    """
    if len(times) == 0:
        return labels, times, times

    if not np.all(times[:-1] <= times[1:]):
        raise ValueError("Input `times` must be sorted in ascending order.")

    d_left = np.timedelta64(max_left_secs, "s")
    d_right = np.timedelta64(max_right_secs, "s")
    start_times = times - d_left
    end_times = times + d_right

    # find where the labels are the same
    labels_same = labels[:-1] == labels[1:]
    # find where the boundaries overlap
    overlap = end_times[:-1] > start_times[1:]

    # do not keep labels that are the same and overlap
    drop = labels_same & overlap
    labels = np.insert(labels[1:][~drop], 0, labels[0])
    start_times = np.insert(start_times[1:][~drop], 0, start_times[0])

    end_times = np.append(end_times[:-1][~drop], end_times[-1])

    # calculate the average of the two boundaries
    dts = (
        (end_times[:-1] - start_times[1:]) + d_right - d_left
    ) / 2  # note that this rounds down to the nearest second
    avg_boundaries = start_times[1:] + dts

    start_times[1:] = np.maximum(start_times[1:], avg_boundaries)
    end_times[:-1] = np.minimum(end_times[:-1], avg_boundaries)

    return (labels, start_times, end_times)

def get_participant_images(base_dir: str, participant_id: str) -> List[str]:
    """Get list of image paths for a participant.
    
    Args:
        base_dir: Base directory containing the dataset
        participant_id: Participant ID
        
    Returns:
        List of image paths
    """
    camera_dir = os.path.join(base_dir, "raw-camera-images", f"camera_participant_{participant_id}")
    return sorted(glob.glob(os.path.join(camera_dir, "*.JPG")))

def get_participant_acc_data(base_dir: str, participant_id: str) -> Optional[str]:
    """Get accelerometer data path for a participant.
    
    Args:
        base_dir: Base directory containing the dataset
        participant_id: Participant ID
        
    Returns:
        Path to accelerometer data file or None if not found
    """
    acc_dir = os.path.join(base_dir, "raw-axivity-accFiles")
    pattern = os.path.join(acc_dir, f"participant_{participant_id}_axivity_data.CWA.gz")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def create_initial_labels(base_dir: str, participant_id: str) -> List[Annotation]:
    """Create initial labels for a participant using blip2 annotations.
    
    Args:
        base_dir: Base directory containing the dataset
        participant_id: Participant ID
        
    Returns:
        List of Annotation objects
    """
    # Load blip2 annotations
    blip2_df = load_blip2_annotations(base_dir)
    
    # Convert participant_id to base form for comparison
    base_id = get_base_participant_id(participant_id)
    
    # Filter for this participant
    participant_df = blip2_df[blip2_df['id'] == base_id].copy()
    
    if len(participant_df) == 0:
        print(f"No blip2 annotations found for participant {participant_id} (base ID: {base_id})")
        return []
    
    # Sort by time
    participant_df.sort_values('time', inplace=True)
    
    # Convert to numpy arrays for spread_label_time
    labels = participant_df['blip2_label'].values
    times = participant_df['time'].values
    
    # Spread labels with 30 second overlap
    _, start_times, end_times = spread_label_time(labels, times, max_left_secs=30, max_right_secs=30)
    
    # Create annotations
    annotations = []
    for label, start, end in zip(labels, start_times, end_times):
        annotations.append(Annotation(
            label=label,
            start_time=start,
            end_time=end
        ))
    
    return annotations

def load_existing_labels(base_dir: str, participant_id: str) -> List[Annotation]:
    """Load existing labels for a participant.
    
    Args:
        base_dir: Base directory containing the dataset
        participant_id: Participant ID
        
    Returns:
        List of Annotation objects
    """
    labels_dir = os.path.join(base_dir, "labels", participant_id)
    annotations = []
    
    # Load each label file
    for label_file in glob.glob(os.path.join(labels_dir, "*.csv")):
        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            annotations.append(Annotation(
                label=row['label'],
                start_time=np.datetime64(row['start_time']),
                end_time=np.datetime64(row['end_time'])
            ))
    
    return annotations


if __name__ == "__main__":
    main()

