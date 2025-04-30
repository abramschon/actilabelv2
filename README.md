# ActiLabel v2

A Python-based tool for annotating time series data and synchronized images. This tool allows you to visualize and annotate multiple data streams simultaneously, making it ideal for labeling sensor data, activity recognition datasets, and other time-synchronized data collections.

## Features

- Multi-channel visualization of time series data (scalar and vector)
- Synchronized image display
- Flexible annotation system with customizable labels
- Interactive timeline navigation
- Zoom and pan controls
- Real-time data visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/actilabelv2.git
cd actilabelv2
```

2. Install required dependencies:
```bash
pip install numpy pygame pillow
```

## Usage

### Running the Example

To try out the tool with example data:

```bash
python example.py
```

This will create a sample dataset with:
- 3-axis accelerometer data (100Hz)
- Temperature readings (1Hz)
- Light sensor readings (1Hz)
- Camera images (0.05Hz / every 20 seconds)
- Two annotation channels: "Location" and "Activity"

### Using Your Own Data

To use the tool with your own data, create instances of the appropriate data source classes:

```python
from main import AnnotationTool, VectorDataSource, ScalarDataSource, ImageDataSource, AnnotationChannel

# Create data sources
accel_source = VectorDataSource(
    "Accelerometer",
    timestamps,  # numpy array of np.datetime64
    data,       # numpy array of shape (n_samples, n_dimensions)
    dim_names=["X", "Y", "Z"]
)

# Create annotation channels
activity_channel = AnnotationChannel(
    "Activity",
    ["walking", "running", "sitting"]  # possible labels
)

# Create and run the tool
tool = AnnotationTool()
tool.load_data([accel_source], [activity_channel])
tool.run()
```

## Controls

### Navigation
- Left/Right Arrow: Move timeline
- =/- or +/-: Zoom in/out
- Space: Start creating a new annotation
- Enter: Finish creating annotation (when editing)
- F1: Toggle help display
- Tab: Switch between annotation channels (when not editing)
- Up/Down: Select channels (when not editing) or navigate through label suggestions (when editing)

### Annotation
1. Press Space to start a new annotation
2. Type the label (autocomplete suggestions will appear)
3. Use Up/Down arrows to navigate suggestions
4. Press Tab to accept suggestion or continue typing
5. Press Enter to create the annotation

## Data Sources

The tool supports three types of data sources:

1. `ScalarDataSource`: For single-value time series (e.g., temperature, light)
2. `VectorDataSource`: For multi-dimensional time series (e.g., accelerometer)
3. `ImageDataSource`: For synchronized images or video frames

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.