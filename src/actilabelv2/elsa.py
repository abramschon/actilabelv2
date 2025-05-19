import re
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
            print(" None")
            return
            
        # Sort participants
        participants = sorted(participants)
        
        # Calculate number of columns (4 columns with 5 char width each)
        col_width = 5
        num_cols = 4
        
        # Print in columns
        for i in range(0, len(participants), num_cols):
            row = participants[i:i + num_cols]
            print(" " + "".join(f"{p:<{col_width}}" for p in row))
    
    print_participant_columns(labeled_participants, "Labeled participants")
    print_participant_columns(unlabeled_participants, "Unlabeled participants")
    
    # Get participant selection
    while True:
        participant_id = input("\nEnter participant ID to label (or 'q' to quit): ").strip()
        if participant_id.lower() == 'q':
            break

        # Try parse to integer
        try:
            participant_id = int(participant_id)
        except ValueError as e:
            print(f"Invalid participant ID: {participant_id}")
            
        # Check if the entered ID exists in either list
        if participant_id not in labeled_participants and participant_id not in unlabeled_participants:
            print(f"Invalid participant ID: {participant_id}")
            continue
        
        # Get image paths
        image_files = get_participant_images(base_dir, participant_id)
        if not image_files:
            print(f"No images found for participant {participant_id}")
            continue
            
        image_times = []
        image_paths = []
        for img_file in image_files:
            timestamp = parse_timestamp_from_filename(
                os.path.basename(img_file), 
                "%Y%m%d_%H%M%S"
            )
            if timestamp:
                image_times.append(timestamp)
                image_paths.append(img_file)
        
        if not image_times:
            print("No valid timestamps found in image filenames")
            continue

        # Create data sources list with both preview and thumbnail image sources
        data_sources = [
            ImageDataSource(
                "Preview",
                np.array(image_times),
                image_paths,
                thumbnail_size=(800, 600),
                max_images_in_view=1,
                display_mode='grid',
            ),
            ImageDataSource(
                "Thumbnails",
                np.array(image_times),
                image_paths,
                thumbnail_size=(100, 75),
                max_images_in_view=10,
                display_mode='centered',
            )
        ]
            
        # Get accelerometer data
        acc_path = get_participant_acc_data(base_dir, participant_id)
        
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
                print(f"Error loading sensor data: {acc_path}")
        else:
            print(f"No accelerometer data found for participant {participant_id}")
        
        # Set up annotation channel
        annotation_channel = AnnotationChannel(
            name="Outdoor",
            possible_labels=["indoor", "outdoor", "uncodeable"]
        )
        
        # Create initial labels if participant is not labelled
        if participant_id not in labeled_participants:
            print(f"Creating initial labels for participant {participant_id}")
            create_initial_labels(base_dir, participant_id)
        
        # Load data into tool
        print("Started annotation tool.")
        # Create annotation tool
        tool = AnnotationTool(
            annotation_dir=get_label_dir(base_dir=base_dir, participant_id=participant_id)
        )

        tool.load_data(
            data_sources=data_sources,
            annotation_channels=[annotation_channel],
            load_existing=True
        )
        
        # Run the tool
        tool.run()

        break # break out of the loop if you have annotated someone


def get_participant_list(base_dir: str) -> Tuple[List[str], List[str]]:
    """Get lists of labeled and unlabeled participants.
    
    Args:
        base_dir: Base directory containing the dataset
        
    Returns:
        Tuple of (labeled_participants, unlabeled_participants)
    """
    # Get all participants with camera images
    camera_dir = os.path.join(base_dir, "raw-camera-images")
    all_participants = []
    for d in os.listdir(camera_dir):
        if d.startswith("camera_participant_"):
            all_participants.append(
                get_base_participant_id(
                    d.split("camera_participant_")[1]
                )
            )
    
    # Get labeled participants
    labels_dir = os.path.join(base_dir, "labels")
    labeled_participants = []
    for d in os.listdir(labels_dir):
        if d.startswith("participant_"):
            labeled_participants.append(
                get_base_participant_id(
                    d.split("participant_")[1]
                )
            )
    
    # Get unlabeled participants
    unlabeled_participants = [p for p in all_participants if p not in labeled_participants]
    
    return labeled_participants, unlabeled_participants

def get_base_participant_id(participant_id: str) -> int:
    """Convert a participant ID to its base form by removing the prefix if present.
    
    Args:
        participant_id: Participant ID (e.g. '1001' or '1')
        
    Returns:
        Base participant ID (e.g. '1' for both '1001' and '1')
    """
    try:
        participant_id = int(participant_id)
    except ValueError as e:
        print("Error parsing", participant_id)
    if participant_id >= 1000:
        participant_id -= 1000
    return participant_id

def get_label_dir(base_dir: str, participant_id: int):
    """
    Creates directories to save each participants annotations within the 
    """
    participant_id += 1000 # to follow the same convention as the other files
    label_dir = os.path.join(base_dir, f"labels/participant_{participant_id}")
    return label_dir 

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
    
    # Initial segment bounds 
    merged_labels = []
    merged_starts = [times[0] - d_left]
    merged_ends = []

    for i in range(len(labels) - 1):
        current_label = labels[i]
        next_label = labels[i+1]
        
        cl_end = times[i] + d_right
        nl_start = times[i+1] - d_left

        if current_label == next_label:
            if cl_end < nl_start:
                merged_labels.append(current_label)
                merged_ends.append(cl_end)
                merged_starts.append(nl_start)
            else: # overlapping
                continue
        else: # different labels
            if cl_end < nl_start:
                merged_labels.append(current_label)
                merged_ends.append(cl_end)
                merged_starts.append(nl_start)
            else: # overlapping
                avg = nl_start + (cl_end - nl_start) / 2
                change = min(avg, times[i+1])
                change = max(change, times[i])
                merged_labels.append(current_label)
                merged_ends.append(change)
                merged_starts.append(change)
    merged_ends.append(times[-1] + d_right)
    merged_labels.append(labels[-1])

    # Convert to arrays
    labels = np.array(merged_labels)
    start_times = np.array(merged_starts)
    end_times = np.array(merged_ends)

    return labels, start_times, end_times


def get_participant_images(base_dir: str, participant_id: int) -> List[str]:
    """Get list of image paths for a participant.
    
    Args:
        base_dir: Base directory containing the dataset
        participant_id: Participant ID
        
    Returns:
        List of image paths
    """
    participant_id = 1000 + participant_id
    camera_dir = os.path.join(base_dir, "raw-camera-images", f"camera_participant_{participant_id}")
    return sorted(glob.glob(os.path.join(camera_dir, "*.JPG")))

def get_participant_acc_data(base_dir: str, participant_id: int) -> Optional[str]:
    """Get accelerometer data path for a participant.
    
    Args:
        base_dir: Base directory containing the dataset
        participant_id: Participant ID
        
    Returns:
        Path to accelerometer data file or None if not found
    """
    participant_id = 1000 + participant_id
    acc_dir = os.path.join(base_dir, "raw-axivity-accFiles")
    # participant_1001_axivity_data.CWA.gz
    acc_path = os.path.join(acc_dir, f"participant_{participant_id}_axivity_data.CWA.gz")
    if os.path.exists(acc_path):
        return acc_path
    
    return None

def create_initial_labels(base_dir: str, participant_id: int) -> bool:
    """Create initial labels for a participant using blip2 annotations.
    
    Args:
        base_dir: Base directory containing the dataset
        participant_id: Participant ID

    Saves the initial labels to Outdoor.csv.
        
    """
    # Load blip2 annotations
    blip2_df = load_blip2_annotations(base_dir)

    blip2_df['blip2_label'] = blip2_df['blip2_label'].replace({'indoors': 'indoor', 'outdoors': 'outdoor'})
    
    # Filter for this participant
    participant_df = blip2_df[blip2_df['id'] == participant_id].copy()
    
    if len(participant_df) == 0:
        print(f"No blip2 annotations found for participant {participant_id} (base ID: {participant_id})")
        return []
    
    # Sort by time
    participant_df.sort_values('time', inplace=True)
    
    # Convert to numpy arrays for spread_label_time
    labels = participant_df['blip2_label'].values
    times = participant_df['time'].values
    
    # Spread labels
    labels, start_times, end_times = spread_label_time(labels, times, max_left_secs=0, max_right_secs=180)
    
    # Save to Outdoor.csv 
    init_df = pd.DataFrame(
        {
            "label": labels,
            "start_time": start_times,
            "end_time": end_times
        }
    )
    label_dir = get_label_dir(base_dir=base_dir, participant_id=participant_id)
    os.mkdir(label_dir)
    label_dir = os.path.join(label_dir, "Outdoor.csv")
    init_df.to_csv(label_dir, index=False)  
    
    return True

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


def parse_timestamp_from_filename(filename, timestamp_format):
    """Extract timestamp from filename using the provided format."""
    try:
        # Get the filename without extension
        name = os.path.splitext(filename)[0]
        
        # Extract the timestamp portion using regex
        # Convert the format pattern to a regex pattern
        # Replace format codes with their regex equivalents
        regex_pattern = timestamp_format.replace('%Y', r'\d{4}')  # 4 digits for year
        regex_pattern = regex_pattern.replace('%m', r'\d{2}')     # 2 digits for month
        regex_pattern = regex_pattern.replace('%d', r'\d{2}')     # 2 digits for day
        regex_pattern = regex_pattern.replace('%H', r'\d{2}')     # 2 digits for hour
        regex_pattern = regex_pattern.replace('%M', r'\d{2}')     # 2 digits for minute
        regex_pattern = regex_pattern.replace('%S', r'\d{2}')     # 2 digits for second
        
        # Find the timestamp in the filename
        match = re.search(regex_pattern, name)
        if not match:
            return None
            
        # Extract the matched timestamp
        timestamp_str = match.group(0)
        
        # Parse the timestamp using the provided format
        dt = datetime.strptime(timestamp_str, timestamp_format)
        return np.datetime64(dt)
    except Exception as e:
        print(f"Error parsing timestamp from {filename}: {e}")
        return None


if __name__ == "__main__":
    main()

