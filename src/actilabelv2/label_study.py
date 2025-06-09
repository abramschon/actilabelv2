import re
import os
import numpy as np
from PIL import Image
import colorsys
from actilabelv2.main import AnnotationTool, VectorDataSource, ScalarDataSource, ImageDataSource, AnnotationChannel
import glob
from datetime import datetime
import argparse
import actipy
import pandas as pd
import yaml


def main():
    """
    Inputs:
    - path to project configuration file
    - path to folder of time-stamped images (optional)
    - path to .CWA file of Axivity AX3 sensor data (optional)
    - path to output folder
    
    Note: At least one of --images or --cwa must be provided.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Wearable Annotation Tool')
    parser.add_argument('--config', '-f',
                      type=str,
                      required=True,
                      help='Path to project configuration YAML file')
    parser.add_argument('--images', '-i', 
                      type=str,
                      required=False,
                      help='Path to the camera input folder containing time-stamped images (optional)')
    parser.add_argument('--cwa', '-c',
                      type=str,
                      required=False,
                      help='Path to the CWA file of sensor data (optional)')
    parser.add_argument('--output', '-o',
                      type=str,
                      required=True,
                      help='Path to the annotation output folder')

    
    args = parser.parse_args()
    
    # Validate that at least one data source is provided
    if not args.images and not args.cwa:
        parser.error("At least one of --images or --cwa must be provided")
    
    # Load project configuration
    try:
        config = load_project_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    data_sources = []
    
    # Process images if provided
    if args.images:
        # Validate input folder exists
        if not os.path.exists(args.images):
            print(f"Error: Input folder '{args.images}' does not exist")
            return
        
        # Get image files
        image_files = []
        for ext in ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.PNG', '*.png']:
            image_files.extend(glob.glob(os.path.join(args.images, ext)))
        
        if not image_files:
            print(f"No image files (.jpg, .jpeg, .png) found in {args.images}")
            return
        
        # Sort files by timestamp
        image_files.sort()
        
        # Extract timestamps and create paths
        image_times = []
        image_paths = []
        for img_file in image_files:
            timestamp = parse_timestamp_from_filename(
                os.path.basename(img_file), 
                config['timestamp_format']['pattern']
            )
            if timestamp:
                image_times.append(timestamp)
                image_paths.append(img_file)
        
        if not image_times:
            print("No valid timestamps found in image filenames")
            return
        
        image_times = np.array(image_times)
        image_times = image_times.astype(np.datetime64)
        
        # Apply time offset if specified in config
        time_offset = config.get('time_offset', "00:00:00")
        if time_offset != "00:00:00":
            try:
                # Parse the time offset string
                # Handle negative offsets
                is_negative = time_offset.startswith('-')
                if is_negative:
                    time_offset = time_offset[1:]  # Remove the minus sign
                
                # Split into hours, minutes, seconds (and optional milliseconds)
                parts = time_offset.split(':')
                if len(parts) != 3:
                    raise ValueError("Time offset must be in format HH:MM:SS or HH:MM:SS.SSS")
                
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])  # This will handle both "SS" and "SS.SSS" formats
                
                # Calculate total seconds
                total_seconds = hours * 3600 + minutes * 60 + seconds
                if is_negative:
                    total_seconds = -total_seconds
                
                # Convert to nanoseconds and apply offset
                offset_ns = int(total_seconds * 1e9)
                image_times = image_times + np.timedelta64(offset_ns, 'ns')
                print(f"Applied time offset of {time_offset} to image timestamps")
            except Exception as e:
                print(f"Error applying time offset: {e}")
                print("Using original timestamps without offset")
        
        # Get display settings from config
        display_settings = config.get('display_settings', {
            'preview': {'size': [300, 300], 'max_images': 1},
            'thumbnails': {'size': [100, 100], 'max_images': 10}
        })
        
        data_sources.extend([
            ImageDataSource(
                "Preview",
                image_times,
                image_paths,
                thumbnail_size=tuple(display_settings['preview']['size']),
                max_images_in_view=display_settings['preview']['max_images'],
            ),
            ImageDataSource(
                "Thumbnails",
                image_times,
                image_paths,
                thumbnail_size=tuple(display_settings['thumbnails']['size']),
                max_images_in_view=display_settings['thumbnails']['max_images'],
                display_mode=display_settings['thumbnails'].get('display_mode', 'centered'),
            ),
        ])

    # Load sensor data using actipy if CWA file is provided
    if args.cwa:
        try:
            print(f"Loading sensor data from {args.cwa}")
            data, info = actipy.read_device(args.cwa,
                                          lowpass_hz=20,
                                          calibrate_gravity=True,
                                          detect_nonwear=True,
                                          resample_hz=50)
            
            # Convert index to numpy datetime64
            sensor_times = data.index.values.astype(np.datetime64)
            
            # Get scalar colors from config if available
            scalar_colors = config.get('scalar_colors', {})

            # Create data sources
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
                    color=tuple(scalar_colors.get('Temperature', [255, 100, 100]))
                ),
                ScalarDataSource(
                    "Light",
                    sensor_times,
                    data['light'].values,
                    color=tuple(scalar_colors.get('Light', [255, 255, 100]))
                )
            ])
        except Exception as e:
            print(f"Error loading sensor data: {e}")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created annotation output folder: {args.output}")
    
    # Create annotation channels from config
    annotation_channels = [
        AnnotationChannel(
            channel['name'],
            channel['labels']
        )
        for channel in config['annotation_channels']
    ]
    
    # Create and run the annotation tool
    tool = AnnotationTool(annotation_dir=args.output)
    tool.load_data(data_sources, annotation_channels, load_existing=True)
    tool.run()


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
        regex_pattern = regex_pattern.replace('%f', r'\d{1,6}')   # 1-6 digits for microseconds
        
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


def load_project_config(config_file):
    """Load project configuration from YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['timestamp_format', 'annotation_channels']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file")
    
    return config




if __name__ == "__main__":
    main() 