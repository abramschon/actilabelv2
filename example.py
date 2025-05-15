import os
import numpy as np
from PIL import Image
import colorsys
from actilabelv2.main import AnnotationTool, VectorDataSource, ScalarDataSource, ImageDataSource, AnnotationChannel

def generate_color_image(size=(640, 480), hue=0.0):
    """Generate a color image with the given hue."""
    # Convert HSV to RGB (saturation and value are fixed)
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
    
    # Create RGB array
    img_array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img_array[:, :] = [int(c * 255) for c in rgb]
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    return img

def create_example_dataset():
    """Create an example dataset with sensor data and images."""
    # Create example_images directory if it doesn't exist
    if not os.path.exists("example_images"):
        os.makedirs("example_images")
    
    # Set up the start time and duration
    start_time = np.datetime64('2024-01-01T09:00:00')  # Start at 9 AM
    duration_seconds = 100 * 20  # 100 images * 20 seconds = 2000 seconds (about 33 minutes)
    
    # Create time arrays for different sensors
    image_times = np.array([
        start_time + np.timedelta64(i * 20, 's')  # Image every 20 seconds
        for i in range(100)
    ])
    
    accel_times = np.array([
        start_time + np.timedelta64(i * 10, 'ms')  # 100Hz accelerometer data
        for i in range(duration_seconds * 100)
    ])
    
    env_times = np.array([
        start_time + np.timedelta64(i, 's')  # 1Hz environmental data
        for i in range(duration_seconds)
    ])
    
    # Generate and save images with gradually changing colors
    image_paths = []
    for i in range(len(image_times)):
        # Calculate hue (cycles through colors twice)
        hue = (i / len(image_times)) * 2 % 1.0
        
        # Generate image
        img = generate_color_image(hue=hue)
        
        # Save image with timestamp in filename
        timestamp = image_times[i].item().strftime('%Y%m%d_%H%M%S')
        filename = f"example_images/image_{timestamp}.jpg"
        img.save(filename)
        image_paths.append(filename)
    
    # Create simulated sensor data
    def generate_activity_pattern(freq, amplitude, noise_level, n_samples):
        t = np.linspace(0, 2*np.pi, n_samples)
        base = amplitude * np.sin(freq * t)
        noise = np.random.normal(0, noise_level, n_samples)
        return base + noise
    
    # Generate different patterns for different activities
    accel_data = np.zeros((len(accel_times), 3))
    activity_segments = 10  # Number of different activity segments
    samples_per_segment = len(accel_times) // activity_segments
    
    for i in range(activity_segments):
        start_idx = i * samples_per_segment
        end_idx = (i + 1) * samples_per_segment
        
        # Randomly choose an activity pattern
        pattern_type = np.random.choice(['sitting', 'walking', 'running', 'standing', 'lying'])
        
        if pattern_type == 'sitting':
            # Low frequency, low amplitude
            accel_data[start_idx:end_idx, 0] = generate_activity_pattern(0.5, 0.2, 0.1, samples_per_segment)
            accel_data[start_idx:end_idx, 1] = generate_activity_pattern(0.5, 0.2, 0.1, samples_per_segment)
            accel_data[start_idx:end_idx, 2] = generate_activity_pattern(0.5, 0.2, 0.1, samples_per_segment)
        elif pattern_type == 'walking':
            # Medium frequency, medium amplitude
            accel_data[start_idx:end_idx, 0] = generate_activity_pattern(2.0, 1.0, 0.2, samples_per_segment)
            accel_data[start_idx:end_idx, 1] = generate_activity_pattern(2.0, 1.0, 0.2, samples_per_segment)
            accel_data[start_idx:end_idx, 2] = generate_activity_pattern(2.0, 0.5, 0.2, samples_per_segment)
        elif pattern_type == 'running':
            # High frequency, high amplitude
            accel_data[start_idx:end_idx, 0] = generate_activity_pattern(4.0, 2.0, 0.3, samples_per_segment)
            accel_data[start_idx:end_idx, 1] = generate_activity_pattern(4.0, 2.0, 0.3, samples_per_segment)
            accel_data[start_idx:end_idx, 2] = generate_activity_pattern(4.0, 1.0, 0.3, samples_per_segment)
        elif pattern_type == 'standing':
            # Very low frequency, very low amplitude
            accel_data[start_idx:end_idx, 0] = generate_activity_pattern(0.1, 0.1, 0.05, samples_per_segment)
            accel_data[start_idx:end_idx, 1] = generate_activity_pattern(0.1, 0.1, 0.05, samples_per_segment)
            accel_data[start_idx:end_idx, 2] = generate_activity_pattern(0.1, 0.1, 0.05, samples_per_segment)
        else:  # lying
            # Almost no movement
            accel_data[start_idx:end_idx, 0] = generate_activity_pattern(0.05, 0.05, 0.02, samples_per_segment)
            accel_data[start_idx:end_idx, 1] = generate_activity_pattern(0.05, 0.05, 0.02, samples_per_segment)
            accel_data[start_idx:end_idx, 2] = generate_activity_pattern(0.05, 0.05, 0.02, samples_per_segment)
    
    # Environmental data
    # Temperature: Gradual changes with some indoor/outdoor transitions
    temperature = np.zeros(len(env_times))
    indoor_temp = 22.0  # 22°C indoor
    outdoor_temp = 15.0  # 15°C outdoor
    is_indoor = True
    
    for i in range(len(env_times)):
        if i % 300 == 0:  # Switch indoor/outdoor every 5 minutes
            is_indoor = not is_indoor
        target_temp = indoor_temp if is_indoor else outdoor_temp
        if i > 0:
            # Gradual temperature change
            temperature[i] = temperature[i-1] + (target_temp - temperature[i-1]) * 0.1
        else:
            temperature[i] = indoor_temp
            
    # Light levels: Correlate with indoor/outdoor transitions
    light = np.zeros(len(env_times))
    indoor_light = 200.0  # 200 lux indoor
    outdoor_light = 10000.0  # 10000 lux outdoor
    
    for i in range(len(env_times)):
        if i % 300 == 0:  # Switch indoor/outdoor every 5 minutes
            is_indoor = not is_indoor
        target_light = indoor_light if is_indoor else outdoor_light
        if i > 0:
            # Gradual light change
            light[i] = light[i-1] + (target_light - light[i-1]) * 0.1
        else:
            light[i] = indoor_light
    
    # Create data sources
    data_sources = [
        ImageDataSource(
            "Preview",
            image_times,
            image_paths,
            thumbnail_size=(300, 300),
            max_images_in_view=1,
        ),
        ImageDataSource(
            "Thumbnail",
            image_times,
            image_paths,
            thumbnail_size=(320, 150),
            max_images_in_view=10,
        ),
        VectorDataSource(
            "Accelerometer",
            accel_times,
            accel_data,
            dim_names=["X", "Y", "Z"]
        ),
        ScalarDataSource(
            "Temperature",
            env_times,
            temperature,
            color=(255, 128, 0)  # Orange for temperature
        ),
        ScalarDataSource(
            "Light",
            env_times,
            light,
            color=(255, 255, 0)  # Yellow for light
        ),
    ]
    
    # Create annotation channels
    annotation_channels = [
        AnnotationChannel(
            "Location",
            ["indoor", "outdoor"]
        ),
        AnnotationChannel(
            "Activity",
            ["sitting", "standing", "walking", "lying", "running"]
        )
    ]
    
    return data_sources, annotation_channels

def main():
    """Run the annotation tool with example data."""
    # Set up annotation directory
    annotation_dir = "example_annotations"
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
        print(f"Created annotation directory: {annotation_dir}")
    
    # Create example dataset
    data_sources, annotation_channels = create_example_dataset()
    
    # Create and run the tool with custom annotation directory
    tool = AnnotationTool(annotation_dir=annotation_dir)
    
    # Load data and existing annotations if available
    tool.load_data(data_sources, annotation_channels, load_existing=True)
    tool.run()

if __name__ == "__main__":
    main() 