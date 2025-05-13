import os
import numpy as np
from PIL import Image
import colorsys
from main import AnnotationTool, VectorDataSource, ScalarDataSource, ImageDataSource, AnnotationChannel
import glob
from datetime import datetime
import argparse


def parse_timestamp_from_filename(filename):
    """Extract timestamp from ELSA image filename format."""
    # Example filename: B...._..._20211109_072834E.JPG
    try:
        # Split by underscore and get the last two parts
        parts = filename.split('_')
        date_part = parts[-2]  # 20211109
        time_part = parts[-1].split('.')[0]  # 072834E
        # Remove the 'E' suffix from the time part
        time_part = time_part[:6]  # 072834
        timestamp_str = date_part + time_part
        dt = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        return np.datetime64(dt)
    except:
        return None


def get_participant_folders(input_folder):
    """Get list of participant folders in the input directory."""
    return [d for d in os.listdir(input_folder) 
            if os.path.isdir(os.path.join(input_folder, d))]


def main():
    """
    Collect all the participant image folders within a given folder.

    The image folder structure is:
    - camera_input_folder/
        - participant_1/
            - B...._..._20211109_072834E.JPG
            - B...._..._20211109_072835E.JPG
            - ...
        - participant_2/
            - B...._..._20211109_072834E.JPG
            - B...._..._20211109_072835E.JPG
            - ...
        - ...

    We want to then create a mirrored folder structure in the supplied annotation_output_folder.
    - annotation_output_folder/
        - participant_1/
            - annotation_channel1.csv
            - annotation_channel2.csv
            - ...
        - participant_2/
            - annotation_channel1.csv
            - annotation_channel2.csv
            - ...
        - ...


    CLI functionality:
    - Prompt the user for the camera_input_folder and annotation_output_folder.
    - Prompt the user which camera input folder to annotate, e.g. participant_1.
    - Create the annotation_output_folder if it doesn't exist.
    - Try parsing in the .JPG files in the camera_input_folder/participant_1/, getting the timestamps from the file names.
    - Launch the annotation tool, with the image paths and timestamps as the data sources.
    - The simple_posture.csv and certainty.csv files define the potential labels for the simple_posture and certainty annotation channels.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='ELSA Study Annotation Tool')
    parser.add_argument('--input', '-i', 
                      type=str,
                      required=True,
                      help='Path to the camera input folder containing participant directories')
    parser.add_argument('--output', '-o',
                      type=str,
                      required=True,
                      help='Path to the annotation output folder')
    parser.add_argument('--participant', '-p',
                      type=str,
                      help='Optional: Specific participant folder to annotate. If not provided, will show selection menu.')
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        return
    
    # Create output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created annotation output folder: {args.output}")
    
    # Get list of participant folders
    participant_folders = get_participant_folders(args.input)
    if not participant_folders:
        print("No participant folders found in the input directory")
        return
    
    # Handle participant selection
    selected_participant = None
    if args.participant:
        if args.participant in participant_folders:
            selected_participant = args.participant
        else:
            print(f"Error: Participant '{args.participant}' not found in input directory")
            print("Available participants:")
            for folder in participant_folders:
                print(f"- {folder}")
            return
    else:
        # Display available participants and prompt for selection
        print("\nAvailable participants:")
        for i, folder in enumerate(participant_folders, 1):
            print(f"{i}. {folder}")
        
        while True:
            try:
                selection = int(input("\nSelect participant number to annotate: "))
                if 1 <= selection <= len(participant_folders):
                    selected_participant = participant_folders[selection - 1]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Get image files for selected participant
    participant_path = os.path.join(args.input, selected_participant)
    image_files = glob.glob(os.path.join(participant_path, "*.JPG"))
    
    if not image_files:
        print(f"No .JPG files found in {participant_path}")
        return
    
    # Sort files by timestamp
    image_files.sort()
    
    # Extract timestamps and create paths
    image_times = []
    image_paths = []
    for img_file in image_files:
        timestamp = parse_timestamp_from_filename(os.path.basename(img_file))
        if timestamp:
            image_times.append(timestamp)
            image_paths.append(img_file)
    
    if not image_times:
        print("No valid timestamps found in image filenames")
        return
    
    # Create participant output folder
    participant_output_folder = os.path.join(args.output, selected_participant)
    if not os.path.exists(participant_output_folder):
        os.makedirs(participant_output_folder)
    
    # Create data sources
    data_sources = [
        ImageDataSource(
            "Camera",
            image_times,
            image_paths
        )
    ]
    
    # Create annotation channels
    annotation_channels = [
        AnnotationChannel(
            "Simple Posture",
            ["lying", "sitting/reclining", "kneeling/crouching", "squatting", 
             "standing", "stepping/intermittent movement", "slow walking; without load",
             "slow walking; with load", "fast walking; without load", "fast walking; with load",
             "ascending stairs", "descending stairs", "running", "cycling",
             "sports; general", "sports; jumping", "sports; skipping", "sports; climbing",
             "sports; crawling", "sports; dancing", "sports; elliptical",
             "sports; stairmaster", "manual labour", "private/obstructed view"]
        ),
        AnnotationChannel(
            "Certainty",
            ["high certainty", "low certainty"]
        )
    ]
    
    # Create and run the annotation tool
    tool = AnnotationTool(annotation_dir=participant_output_folder)
    tool.load_data(data_sources, annotation_channels, load_existing=True)
    tool.run()


if __name__ == "__main__":
    main() 