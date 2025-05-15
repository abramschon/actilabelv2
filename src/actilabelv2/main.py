import os
import json
import time
import math
import threading
import pygame
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable, Union, Set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime

# Helper to convert hex to RGB
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Constants - Dark Theme Palette
BACKGROUND_COLOR = hex_to_rgb("#282C34") # Dark Bluish Grey
TEXT_COLOR = hex_to_rgb("#ABB2BF")       # Light Grey
GRID_COLOR = hex_to_rgb("#3A3F4B")       # Subtle Grey Grid/Borders
HIGHLIGHT_COLOR = hex_to_rgb("#61AFEF")  # Bright Blue for selections/cursor
ERROR_COLOR = hex_to_rgb("#E06C75")      # Soft Red/Pink
SUCCESS_COLOR = hex_to_rgb("#98C379")    # Soft Green
WARNING_COLOR = hex_to_rgb("#E5C07B")    # Soft Yellow/Gold
CURSOR_COLOR = hex_to_rgb("#5C6370")     # Brighter grey for cursor and timestamp lines
ANNOTATION_COLORS = [
    hex_to_rgb("#61AFEF"),   # Blue
    hex_to_rgb("#98C379"),   # Green
    hex_to_rgb("#C678DD"),   # Purple
    hex_to_rgb("#E5C07B"),   # Yellow/Gold
    hex_to_rgb("#E06C75"),   # Red/Pink
    hex_to_rgb("#56B6C2"),   # Teal
    hex_to_rgb("#D19A66"),   # Orange
    hex_to_rgb("#ABB2BF"),   # Light Grey (same as text)
]
# Define a specific text color for annotations for better contrast
ANNOTATION_TEXT_COLOR = hex_to_rgb("#FFFFFF") # White
DEFAULT_ALPHA = 0.7 # Slightly less alpha might look better on dark theme
FPS = 60
SELECTED_CHANNEL_COLOR = hex_to_rgb("#323842")  # Slightly lighter than background

class TimeScale:
    """Manages time scale conversions between datetime and pixel coordinates."""
    
    def __init__(self, min_time: np.datetime64, max_time: np.datetime64):
        self.min_time = min_time
        self.max_time = max_time
        self.initial_min_time = min_time  # Store initial range
        self.initial_max_time = max_time
        self.update_scale()
        
    def update_scale(self):
        """Update scale calculation."""
        self.scale = max((self.max_time - self.min_time).astype('timedelta64[ms]').astype(int), 1)
        
    def to_unit(self, times: np.ndarray) -> np.ndarray:
        """Convert datetime64 to normalized [0,1] coordinates."""
        delta = (times - self.min_time).astype('timedelta64[ms]').astype(int)
        return delta / self.scale if self.scale else np.zeros_like(delta)
    
    def to_scale(self, units: Union[float, np.ndarray]) -> np.ndarray:
        """Convert normalized [0,1] coordinates to datetime64."""
        if isinstance(units, (int, float)):
            ms = int(units * self.scale)
            return self.min_time + np.timedelta64(ms, 'ms')
        else:
            ms = (units * self.scale).astype(int)
            return self.min_time + np.array([np.timedelta64(m, 'ms') for m in ms])
            
    def time_to_str(self, time: np.datetime64, format: str = "%Y-%m-%d %H:%M:%S.%f") -> str:
        """Convert datetime64 to string."""
        return np.datetime_as_string(time, unit='ms').replace('T', ' ')
    
    def zoom(self, factor: float, center: float):
        """Zoom the time scale by factor around center point (0-1)."""
        if factor <= 0:
            return
            
        center_time = self.to_scale(center)
        range_ms = (self.max_time - self.min_time).astype('timedelta64[ms]').astype(int)
        new_range_ms = int(range_ms / factor)
        
        # Calculate new min and max times while keeping center fixed
        center_offset = int(center * new_range_ms)
        self.min_time = center_time - np.timedelta64(center_offset, 'ms')
        self.max_time = self.min_time + np.timedelta64(new_range_ms, 'ms')
        self.update_scale()
        
    def pan(self, delta: float):
        """Pan the time scale by delta (in normalized units)."""
        time_delta = np.timedelta64(int(delta * self.scale), 'ms')
        self.min_time -= time_delta
        self.max_time -= time_delta
        self.update_scale()
    
    def reset_view(self):
        """Reset view to initial time range."""
        self.min_time = self.initial_min_time
        self.max_time = self.initial_max_time
        self.update_scale()

@dataclass
class Annotation:
    """Represents a single labeled time segment."""
    label: str
    start_time: np.datetime64
    end_time: np.datetime64
    
    def contains_time(self, time: np.datetime64) -> bool:
        """Check if this annotation contains the given time."""
        return self.start_time <= time < self.end_time
    
    def duration(self) -> np.timedelta64:
        """Get annotation duration."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            'label': self.label,
            'start_time': str(self.start_time),
            'end_time': str(self.end_time)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Annotation':
        """Create annotation from dictionary."""
        return cls(
            label=data['label'],
            start_time=np.datetime64(data['start_time']),
            end_time=np.datetime64(data['end_time'])
        )

class AnnotationChannel:
    """Manages a set of temporal annotations for a specific aspect."""
    
    def __init__(self, 
                 name: str, 
                 possible_labels: List[str],
                 color_map: Dict[str, Tuple[int, int, int]] = None):
        self.name = name
        self.possible_labels = possible_labels
        self.annotations: List[Annotation] = []
        
        # Set up colors for the labels
        self.color_map = {}
        if color_map:
            self.color_map = color_map
        else:
            for i, label in enumerate(possible_labels):
                self.color_map[label] = ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)]
    
    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation to this channel."""
        # Check for overlaps and resolve if needed
        print(f"Adding annotation: {annotation.label} ({annotation.start_time} - {annotation.end_time})")
        overlaps = [a for a in self.annotations 
                    if (a.start_time < annotation.end_time and 
                        annotation.start_time < a.end_time)]
        
        # Remove overlaps
        for overlap in overlaps:
            print(f"Removing overlapping annotation: {overlap.label} ({overlap.start_time} - {overlap.end_time})")
            self.annotations.remove(overlap)
            
        self.annotations.append(annotation)
        print(f"Added annotation. New count: {len(self.annotations)}")
        self.sort_annotations()
    
    def remove_annotation(self, annotation: Annotation) -> None:
        """Remove an annotation from this channel."""
        if annotation in self.annotations:
            self.annotations.remove(annotation)
            print(f"Removed annotation. New count: {len(self.annotations)}")
    
    def sort_annotations(self) -> None:
        """Sort annotations by start time."""
        self.annotations.sort(key=lambda a: a.start_time)
    
    def get_annotations_in_range(self, start_time: np.datetime64, end_time: np.datetime64) -> List[Annotation]:
        """Get all annotations that overlap with the given time range."""
        return [a for a in self.annotations if a.end_time > start_time and a.start_time < end_time]
    
    def get_annotation_at_time(self, time: np.datetime64) -> Optional[Annotation]:
        """Get the annotation that contains the given time."""
        for annotation in self.annotations:
            if annotation.contains_time(time):
                return annotation
        return None
    
    def find_gaps(self, start_time: np.datetime64, end_time: np.datetime64) -> List[Tuple[np.datetime64, np.datetime64]]:
        """Find gaps in annotations between start_time and end_time."""
        annotations = self.get_annotations_in_range(start_time, end_time)
        annotations.sort(key=lambda a: a.start_time)
        
        gaps = []
        current = start_time
        
        for a in annotations:
            if a.start_time > current:
                gaps.append((current, a.start_time))
            current = max(current, a.end_time)
        
        if current < end_time:
            gaps.append((current, end_time))
            
        return gaps
    
    def add_label(self, label: str) -> None:
        """Add a new label to the possible labels if it doesn't exist."""
        if label not in self.possible_labels:
            self.possible_labels.append(label)
            # Assign a color
            if label not in self.color_map:
                idx = len(self.possible_labels) - 1
                self.color_map[label] = ANNOTATION_COLORS[idx % len(ANNOTATION_COLORS)]
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            'name': self.name,
            'possible_labels': self.possible_labels,
            'annotations': [a.to_dict() for a in self.annotations],
            'color_map': {k: list(v) for k, v in self.color_map.items()}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnnotationChannel':
        """Create channel from dictionary."""
        channel = cls(
            name=data['name'],
            possible_labels=data['possible_labels']
        )
        # Restore color map if present
        if 'color_map' in data:
            channel.color_map = {k: tuple(v) for k, v in data['color_map'].items()}
        
        # Restore annotations
        channel.annotations = [Annotation.from_dict(a) for a in data['annotations']]
        return channel
    
    def render(self, surface: pygame.Surface, time_scale: Optional[TimeScale], rect: pygame.Rect) -> None:
        """Render annotations in the given rectangle."""
        if time_scale is None:
            # Draw "No data" text
            font = pygame.font.SysFont("Helvetica Neue", 20)
            text = font.render("No timeline data loaded", True, TEXT_COLOR)
            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
            return
            
        # Find annotations in the visible range
        visible_annotations = [
            a for a in self.annotations 
            if a.end_time > time_scale.min_time and a.start_time < time_scale.max_time
        ]
        
        if not visible_annotations:
            # Draw "No data" text
            font = pygame.font.SysFont("Helvetica Neue", 20)
            text = font.render("No annotations in view", True, TEXT_COLOR)
            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
            return
            
        # Calculate annotation height (leave some margin at top and bottom)
        annotation_height = rect.height - 20  # 10px margin top and bottom
        annotation_y = rect.top + 10  # Start 10px from top
        
        # Draw each annotation as a colored rectangle
        for annotation in visible_annotations:
            # Convert times to unit scale
            start_unit = max(0, time_scale.to_unit(np.array([annotation.start_time]))[0])
            end_unit = min(1, time_scale.to_unit(np.array([annotation.end_time]))[0])
            
            # Calculate rectangle coordinates
            x1 = max(rect.left, rect.left + start_unit * rect.width)
            x2 = min(rect.right, rect.left + end_unit * rect.width)
            
            # Skip if too small to be visible
            if x2 - x1 < 1:
                continue
                
            # Get color for this label
            color = self.color_map.get(annotation.label, TEXT_COLOR)
            
            # Draw the rectangle with alpha blending
            width = max(1, x2 - x1)  # Ensure width is at least 1
            s = pygame.Surface((width, annotation_height))
            s.set_alpha(int(DEFAULT_ALPHA * 255))
            s.fill(color)
            surface.blit(s, (x1, annotation_y))
            
            # Draw border
            pygame.draw.rect(surface, color, (x1, annotation_y, width, annotation_height), 1)
            
            # Draw label text if there's enough space
            if width > 30:
                font = pygame.font.SysFont("Helvetica Neue", 14) # Reduced size
                # Use the dedicated ANNOTATION_TEXT_COLOR for readability
                text = font.render(annotation.label, True, ANNOTATION_TEXT_COLOR) 
                text_x = x1 + (width // 2) - text.get_width() // 2
                text_y = rect.centery - text.get_height() // 2
                
                # Draw the text directly (no background rectangle needed with good contrast)
                surface.blit(text, (text_x, text_y))
            else:
                # Draw tiny indicator (maybe less necessary now, but keep for very small segments)
                # Ensure indicator color contrasts with background
                indicator_color = ANNOTATION_TEXT_COLOR # Use white indicator
                pygame.draw.rect(surface, indicator_color, (x1, rect.top + 5, max(1, width), 3), 0)

class DataSource:
    """Base class for different data sources."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_time_range(self) -> Tuple[np.datetime64, np.datetime64]:
        """Get the time range of this data source."""
        raise NotImplementedError()
    
    def render(self, surface: pygame.Surface, time_scale: Optional[TimeScale], rect: pygame.Rect) -> None:
        """Render this data source on the given surface."""
        if time_scale is None:
            font = pygame.font.SysFont("Helvetica Neue", 20)
            text = font.render(f"No time data loaded for {self.name}", True, TEXT_COLOR)
            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
            return
        raise NotImplementedError()

class ScalarDataSource(DataSource):
    """A data source for scalar values over time."""
    
    def __init__(self, name: str, times: np.ndarray, values: np.ndarray, color: Tuple[int, int, int] = (255, 255, 255)):
        super().__init__(name)
        self.times = times
        self.values = values
        self.color = color
        # Precompute min/max for efficient rendering
        self.min_value = np.min(values) if len(values) else 0
        self.max_value = np.max(values) if len(values) else 1
        self.range = max(self.max_value - self.min_value, 1e-9)  # Avoid division by zero
        self.y_scale = 1.0  # Add y-axis scale factor
    
    def get_time_range(self) -> Tuple[np.datetime64, np.datetime64]:
        if len(self.times) == 0:
            return np.datetime64('2000-01-01'), np.datetime64('2000-01-02')
        return self.times[0], self.times[-1]
    
    def scale_y(self, factor: float):
        """Scale the y-axis by the given factor."""
        if factor <= 0:
            return
        self.y_scale *= factor
        # Ensure scale stays within reasonable bounds
        self.y_scale = max(0.1, min(10.0, self.y_scale))
    
    def render(self, surface: pygame.Surface, time_scale: Optional[TimeScale], rect: pygame.Rect) -> None:
        if time_scale is None:
            super().render(surface, time_scale, rect)
            return
            
        # Get points in the visible range
        visible_mask = np.logical_and(
            self.times >= time_scale.min_time,
            self.times <= time_scale.max_time
        )
        visible_times = self.times[visible_mask]
        visible_values = self.values[visible_mask]
        
        if len(visible_times) < 2:
            font = pygame.font.SysFont("Helvetica Neue", 20)
            text = font.render(f"No data in view for {self.name}", True, TEXT_COLOR)
            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
            return
            
        # Convert to screen coordinates
        unit_times = time_scale.to_unit(visible_times)
        x_coords = rect.left + unit_times * rect.width
        
        # Calculate scaled range for y-axis
        center_value = (self.min_value + self.max_value) / 2
        scaled_range = self.range / self.y_scale
        scaled_min = center_value - scaled_range / 2
        scaled_max = center_value + scaled_range / 2
        
        # Normalize values to the rect height using scaled range
        norm_values = (visible_values - scaled_min) / (scaled_max - scaled_min)
        y_coords = rect.bottom - norm_values * (rect.height - 20) - 10  # Leave margin
        
        # Draw the line
        points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
        if len(points) > 1:
            pygame.draw.lines(surface, self.color, False, points, 1)
            
        # Draw axis labels with smaller font
        font = pygame.font.SysFont("Helvetica Neue", 12)  # Reduced from 16 to 12
        min_label = font.render(f"{scaled_min:.2f}", True, TEXT_COLOR)
        max_label = font.render(f"{scaled_max:.2f}", True, TEXT_COLOR)
        surface.blit(min_label, (rect.left + 5, rect.bottom - 20))
        surface.blit(max_label, (rect.left + 5, rect.top + 5))

class VectorDataSource(DataSource):
    """A data source for vector values over time."""
    
    def __init__(self, name: str, times: np.ndarray, values: np.ndarray, 
                 dim_names: List[str] = None, colors: List[Tuple[int, int, int]] = None):
        super().__init__(name)
        self.times = times
        self.values = values  # Shape: [n_times, n_dimensions]
        self.n_dims = values.shape[1] if values.ndim > 1 else 1
        
        # Set dimension names
        if dim_names is None:
            self.dim_names = [f"{name}_{i}" for i in range(self.n_dims)]
        else:
            self.dim_names = dim_names[:self.n_dims]
            
        # Set colors for each dimension
        if colors is None:
            self.colors = [ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)] for i in range(self.n_dims)]
        else:
            self.colors = colors[:self.n_dims]
            
        # Precompute min/max for efficient rendering
        self.min_value = np.min(values) if len(values) else 0
        self.max_value = np.max(values) if len(values) else 1
        self.range = max(self.max_value - self.min_value, 1e-9)  # Avoid division by zero
        self.y_scale = 1.0  # Add y-axis scale factor
    
    def get_time_range(self) -> Tuple[np.datetime64, np.datetime64]:
        if len(self.times) == 0:
            return np.datetime64('2000-01-01'), np.datetime64('2000-01-02')
        return self.times[0], self.times[-1]
    
    def scale_y(self, factor: float):
        """Scale the y-axis by the given factor."""
        if factor <= 0:
            return
        self.y_scale *= factor
        # Ensure scale stays within reasonable bounds
        self.y_scale = max(0.1, min(10.0, self.y_scale))
    
    def render(self, surface: pygame.Surface, time_scale: Optional[TimeScale], rect: pygame.Rect) -> None:
        if time_scale is None:
            super().render(surface, time_scale, rect)
            return
            
        # Get points in the visible range
        visible_mask = np.logical_and(
            self.times >= time_scale.min_time,
            self.times <= time_scale.max_time
        )
        visible_times = self.times[visible_mask]
        visible_values = self.values[visible_mask]
        
        if len(visible_times) < 2:
            font = pygame.font.SysFont("Helvetica Neue", 20)
            text = font.render(f"No data in view for {self.name}", True, TEXT_COLOR)
            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
            return
            
        # Convert to screen coordinates
        unit_times = time_scale.to_unit(visible_times)
        x_coords = rect.left + unit_times * rect.width
        
        # Calculate scaled range for y-axis
        center_value = (self.min_value + self.max_value) / 2
        scaled_range = self.range / self.y_scale
        scaled_min = center_value - scaled_range / 2
        scaled_max = center_value + scaled_range / 2
        
        # Draw each dimension
        for dim in range(self.n_dims):
            # Get values for this dimension
            dim_values = visible_values[:, dim] if visible_values.ndim > 1 else visible_values
            
            # Normalize values to the rect height using scaled range
            norm_values = (dim_values - scaled_min) / (scaled_max - scaled_min)
            y_coords = rect.bottom - norm_values * (rect.height - 20) - 10  # Leave margin
            
            # Draw the line
            points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
            if len(points) > 1:
                pygame.draw.lines(surface, self.colors[dim], False, points, 1)
        
        # Draw legend with smaller font
        font = pygame.font.SysFont("Helvetica Neue", 12)  # Reduced from 16 to 12
        legend_x = rect.left + 10
        legend_y = rect.top + 5
        for dim in range(self.n_dims):
            # Draw color indicator
            pygame.draw.line(surface, self.colors[dim], 
                            (legend_x, legend_y), 
                            (legend_x + 20, legend_y), 
                            2)
            # Draw dimension name
            text = font.render(self.dim_names[dim], True, TEXT_COLOR)
            surface.blit(text, (legend_x + 25, legend_y - text.get_height() // 2))
            legend_y += 20
            
        # Draw y-axis labels
        min_label = font.render(f"{scaled_min:.2f}", True, TEXT_COLOR)
        max_label = font.render(f"{scaled_max:.2f}", True, TEXT_COLOR)
        surface.blit(min_label, (rect.left + 5, rect.bottom - 20))
        surface.blit(max_label, (rect.left + 5, rect.top + 5))

class ImageDataSource(DataSource):
    """A data source for images over time."""
    
    def __init__(self, 
                 name: str, 
                 times: np.ndarray, 
                 image_paths: List[str], 
                 thumbnail_size: Tuple[int, int] = (100, 100),
                 max_images_in_view: int = 10,
                 display_mode: str = "grid"):  # Add display_mode parameter
        super().__init__(name)
        self.times = times
        self.image_paths = image_paths
        self.max_images_in_view = max_images_in_view
        self.display_mode = display_mode  # Store display mode
        self.buffer_images = 2  # Number of images to show in buffer on each side
        self.target_thumbnail_size = thumbnail_size  # Store target size for reference
        
        # Calculate initial aspect ratio from first image
        self.aspect_ratio = 1.0  # Default to square
        if image_paths:
            try:
                with Image.open(image_paths[0]) as img:
                    width, height = img.size
                    self.aspect_ratio = width / height
            except Exception as e:
                print(f"Error loading image for aspect ratio: {e}")
        
        # LRU Cache implementation
        self.thumbnail_cache_size = 100  # Maximum number of thumbnails to keep in memory
        self.thumbnails = {}  # Dictionary to store thumbnails
        self.thumbnail_order = []  # List to track LRU order
        self.thumbnail_locks = {}  # Dictionary to track loading status
        
        # Preload initial thumbnails in background
        self.preload_count = min(20, len(image_paths))
        self.load_thread = threading.Thread(target=self._preload_thumbnails, daemon=True)
        self.load_thread.start()

    def _calculate_image_dimensions(self, rect_width: int, img_margin: int) -> Tuple[int, int]:
        """Calculate image dimensions based on available space and max_images_in_view."""
        # Add extra space on the right for the mode toggle button
        button_width = 50  # Space for mode toggle button
        available_width = rect_width - button_width
        
        # Calculate maximum width based on available space and number of images
        max_width = (available_width - (self.max_images_in_view + 1) * img_margin) // self.max_images_in_view
        
        # Calculate height based on aspect ratio
        if self.aspect_ratio > 1:  # Wider than tall
            width = min(max_width, self.target_thumbnail_size[0])
            height = int(width / self.aspect_ratio)
        else:  # Taller than wide
            height = min(self.target_thumbnail_size[1], 
                        int(max_width / self.aspect_ratio))
            width = int(height * self.aspect_ratio)
        
        return width, height

    def get_height(self) -> int:
        """Get the height needed for this channel based on image dimensions."""
        # Calculate image dimensions based on a default width (will be adjusted in render)
        width, height = self._calculate_image_dimensions(800, 5)  # Use reasonable default width
        
        # Add space for buttons at top, timeline, and connection lines
        top_margin = 40  # Space for buttons at top
        timeline_height = 40  # Height for timeline
        connection_height = 40  # Extra space for connection lines
        return height + top_margin + timeline_height + connection_height

    def _preload_thumbnails(self):
        """Preload thumbnails in background thread."""
        for i, path in enumerate(self.image_paths[:self.preload_count]):
            self._load_thumbnail(path)
    
    def _load_thumbnail(self, path: str) -> bool:
        """Load a thumbnail into the cache with LRU eviction."""
        # If already loading, wait for it
        if path in self.thumbnail_locks:
            return True
            
        # If already in cache, update LRU order
        if path in self.thumbnails:
            self.thumbnail_order.remove(path)
            self.thumbnail_order.append(path)
            return True
        
        try:
            # Mark as loading
            self.thumbnail_locks[path] = True
            
            # Load and process image
            img = Image.open(path)
            
            # Calculate new dimensions while maintaining aspect ratio
            width, height = img.size
            aspect_ratio = width / height
            
            if width > height:
                # Image is wider than tall
                new_width = self.target_thumbnail_size[0]
                new_height = int(new_width / aspect_ratio)
            else:
                # Image is taller than wide
                new_height = self.target_thumbnail_size[1]
                new_width = int(new_height * aspect_ratio)
            
            # Resize image maintaining aspect ratio
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_data = img.convert("RGB")
            img_surface = pygame.image.frombuffer(
                img_data.tobytes(), img_data.size, img_data.mode)
            
            # Implement LRU cache eviction
            if len(self.thumbnails) >= self.thumbnail_cache_size:
                oldest = self.thumbnail_order.pop(0)
                del self.thumbnails[oldest]
            
            # Add to cache
            self.thumbnails[path] = img_surface
            self.thumbnail_order.append(path)
            
            # Remove loading lock
            del self.thumbnail_locks[path]
            return True
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            if path in self.thumbnail_locks:
                del self.thumbnail_locks[path]
            return False
    
    def get_thumbnail(self, path: str) -> Optional[pygame.Surface]:
        """Get a thumbnail for the given path, loading if necessary."""
        # If in cache, update LRU order and return
        if path in self.thumbnails:
            self.thumbnail_order.remove(path)
            self.thumbnail_order.append(path)
            return self.thumbnails[path]
        
        # If not in cache, try to load it
        if self._load_thumbnail(path):
            return self.thumbnails.get(path)  # Use .get() to safely access the thumbnail
        
        return None
    
    def _select_balanced_images(self, indices: np.ndarray, n_images: int) -> np.ndarray:
        """Select n_images evenly distributed across the time range.
        
        Args:
            indices: Array of image indices in the visible range
            n_images: Number of images to select
            
        Returns:
            Array of selected indices
        """
        if len(indices) <= n_images:
            return indices
            
        # Calculate the time chunks
        chunk_size = len(indices) / (n_images + 1)
        
        # Select the middle point of each chunk
        selected_indices = []
        for i in range(n_images):
            # Calculate the middle point of this chunk
            chunk_middle = int((i + 0.5) * chunk_size)
            selected_indices.append(indices[chunk_middle])
            
        return np.array(selected_indices)

    def render(self, surface: pygame.Surface, time_scale: Optional[TimeScale], rect: pygame.Rect) -> None:
        """Render the image data source."""
        if time_scale is None:
            super().render(surface, time_scale, rect)
            return
            
        # Find images in the visible time range
        visible_mask = np.logical_and(
            self.times >= time_scale.min_time,
            self.times <= time_scale.max_time
        )
        visible_indices = np.where(visible_mask)[0]
        
        if len(visible_indices) == 0:
            font = pygame.font.SysFont("Helvetica Neue", 20)
            text = font.render(f"No images in view for {self.name}", True, TEXT_COLOR)
            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
            return
            
        # Get center time (cursor position)
        center_time = time_scale.to_scale(0.5)
        
        # Handle preview vs thumbnail display
        if self.max_images_in_view == 1:  # Preview channel
            # Find the image closest to center time
            if len(visible_indices) > 0:
                time_diffs = np.abs(self.times[visible_indices] - center_time)
                closest_idx = visible_indices[np.argmin(time_diffs)]
                display_indices = [closest_idx]
            else:
                display_indices = visible_indices
        else:  # Thumbnail channel
            if self.display_mode == "grid":
                # Use balanced selection for grid mode
                display_indices = self._select_balanced_images(visible_indices, self.max_images_in_view)
            else:  # centered mode
                # Find images in visible range plus buffer on both sides
                buffer_time = (time_scale.max_time - time_scale.min_time) * 0.2  # 20% buffer on each side
                buffered_min_time = time_scale.min_time - buffer_time
                buffered_max_time = time_scale.max_time + buffer_time
                
                buffered_mask = np.logical_and(
                    self.times >= buffered_min_time,
                    self.times <= buffered_max_time
                )
                buffered_indices = np.where(buffered_mask)[0]
                
                # Always show a fixed number of images (visible + buffer)
                total_images = self.max_images_in_view + (2 * self.buffer_images)
                
                if len(buffered_indices) > total_images:
                    # Use balanced selection for buffered images too
                    display_indices = self._select_balanced_images(buffered_indices, total_images)
                else:
                    display_indices = buffered_indices
        
        # Calculate layout
        img_margin = 5
        top_margin = 40  # Space for buttons at top
        timeline_height = 40  # Height for timeline
        connection_height = 40  # Extra space for connection lines
        img_area_height = rect.height - top_margin - timeline_height - connection_height
        
        # Calculate image dimensions based on available space
        img_width, img_height = self._calculate_image_dimensions(rect.width, img_margin)
        
        # Calculate x positions for images based on display mode
        if len(display_indices) == 1:
            # Center single image
            x_pos = rect.centerx - img_width // 2
            image_positions = [x_pos]
        else:
            if self.display_mode == "grid":
                # Calculate evenly spaced x positions for multiple images
                spacing = (rect.width - img_width - 50) / (len(display_indices) - 1) if len(display_indices) > 1 else 0
                image_positions = [rect.left + i * spacing for i in range(len(display_indices))]
            else:  # centered mode
                # Position images centered over their timestamps
                image_positions = []
                for idx in display_indices:
                    time = self.times[idx]
                    time_unit = time_scale.to_unit(np.array([time]))[0]
                    time_x = rect.left + time_unit * rect.width
                    image_positions.append(time_x - img_width // 2)
        
        # Draw images and timestamps
        for i, (idx, x_pos) in enumerate(zip(display_indices, image_positions)):
            time = self.times[idx]
            path = self.image_paths[idx]
            
            # Calculate image position
            y = rect.top + top_margin + (img_area_height - img_height) // 2
            
            # Calculate alpha based on position relative to visible window
            alpha = 255
            if self.display_mode == "centered":
                time_unit = time_scale.to_unit(np.array([time]))[0]
                # Calculate distance from center of visible window
                center_unit = 0.5
                distance = abs(time_unit - center_unit)
                # Fade out images that are in the buffer zone
                if distance > 0.5:  # Outside visible window
                    alpha = int(255 * (1 - (distance - 0.5) * 2))  # Fade out in buffer zone
                    alpha = max(0, min(255, alpha))  # Clamp between 0 and 255
            
            # Draw the image if loaded
            thumbnail = self.get_thumbnail(path)
            if thumbnail:
                # Scale thumbnail if needed
                if thumbnail.get_width() != img_width or thumbnail.get_height() != img_height:
                    thumbnail = pygame.transform.scale(thumbnail, (img_width, img_height))
                
                # Create a surface with alpha
                if alpha < 255:
                    thumbnail = thumbnail.copy()
                    thumbnail.set_alpha(alpha)
                
                surface.blit(thumbnail, (x_pos, y))
            else:
                # Draw placeholder with alpha
                placeholder_rect = pygame.Rect(x_pos, y, img_width, img_height)
                placeholder_surface = pygame.Surface((img_width, img_height))
                placeholder_surface.set_alpha(alpha)
                placeholder_surface.fill(GRID_COLOR)
                surface.blit(placeholder_surface, (x_pos, y))
                pygame.draw.line(surface, GRID_COLOR, 
                                (placeholder_rect.left, placeholder_rect.top),
                                (placeholder_rect.right, placeholder_rect.bottom), 1)
                pygame.draw.line(surface, GRID_COLOR, 
                                (placeholder_rect.left, placeholder_rect.bottom),
                                (placeholder_rect.right, placeholder_rect.top), 1)
            
            # Calculate actual time position on timeline
            time_unit = time_scale.to_unit(np.array([time]))[0]
            time_x = rect.left + time_unit * rect.width
            
            # Draw connection line from image to timestamp only in grid mode
            if self.display_mode == "grid":
                timeline_y = rect.bottom - timeline_height // 2
                pygame.draw.line(surface, CURSOR_COLOR,
                               (x_pos + img_width // 2, y + img_height),  # Bottom center of image
                               (time_x, timeline_y),  # Timeline position
                               1)
            
            # Draw timestamp marker and label only if in visible range or grid mode
            if self.display_mode == "grid" or (time_unit >= 0 and time_unit <= 1):
                timeline_y = rect.bottom - timeline_height // 2
                pygame.draw.circle(surface, CURSOR_COLOR, (int(time_x), timeline_y), 3)
                
                # Draw time label with smaller font
                font = pygame.font.SysFont("Helvetica Neue", 12)
                time_text = time_scale.time_to_str(time)[11:19]  # Just HH:MM:SS
                text = font.render(time_text, True, TEXT_COLOR)
                text_x = max(rect.left, min(rect.right - text.get_width(),
                                          time_x - text.get_width() // 2))
                surface.blit(text, (text_x, timeline_y + 5))
        
        # Draw timeline
        timeline_y = rect.bottom - timeline_height // 2
        pygame.draw.line(surface, GRID_COLOR,
                        (rect.left, timeline_y),
                        (rect.right, timeline_y),
                        1)

    def get_time_range(self) -> Tuple[datetime, datetime]:
        """Get the time range of the data source."""
        if len(self.times) == 0:
            raise ValueError("No times available in ImageDataSource")
        return self.times[0], self.times[-1]

class TextRangeDataSource(DataSource):
    """A data source for existing annotations."""
    
    def __init__(self, name: str, annotations: List[Annotation], color_map: Dict[str, Tuple[int, int, int]]):
        super().__init__(name)
        self.annotations = annotations
        self.color_map = color_map
    
    def get_time_range(self) -> Tuple[np.datetime64, np.datetime64]:
        if not self.annotations:
            return np.datetime64('2000-01-01'), np.datetime64('2000-01-02')
        return min(a.start_time for a in self.annotations), max(a.end_time for a in self.annotations)
    
    def render(self, surface: pygame.Surface, time_scale: Optional[TimeScale], rect: pygame.Rect) -> None:
        if time_scale is None:
            super().render(surface, time_scale, rect)
            return
            
        # Find annotations in the visible range
        visible_annotations = [
            a for a in self.annotations 
            if a.end_time > time_scale.min_time and a.start_time < time_scale.max_time
        ]
        
        if not visible_annotations:
            # Draw "No data" text
            font = pygame.font.SysFont("Helvetica Neue", 18) # Reduced size
            text = font.render("No annotations in view", True, TEXT_COLOR)
            surface.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
            return
            
        # Draw each annotation as a colored rectangle
        for annotation in visible_annotations:
            # Convert times to unit scale
            start_unit = max(0, time_scale.to_unit(np.array([annotation.start_time]))[0])
            end_unit = min(1, time_scale.to_unit(np.array([annotation.end_time]))[0])
            
            # Calculate rectangle coordinates
            x1 = max(rect.left, rect.left + start_unit * rect.width)
            x2 = min(rect.right, rect.left + end_unit * rect.width)
            
            # Skip if too small to be visible
            if x2 - x1 < 1:
                continue
                
            # Get color for this label
            color = self.color_map.get(annotation.label, TEXT_COLOR)
            
            # Draw the rectangle with alpha blending
            s = pygame.Surface((x2 - x1, rect.height - 10))
            s.set_alpha(int(DEFAULT_ALPHA * 255))
            s.fill(color)
            surface.blit(s, (x1, rect.top + 5))
            
            # Draw border
            pygame.draw.rect(surface, color, (x1, rect.top + 5, x2 - x1, rect.height - 10), 1)
            
            # Draw label text if there's enough space
            if x2 - x1 > 30:
                font = pygame.font.SysFont("Helvetica Neue", 16) # Reduced size
                # Use the dedicated ANNOTATION_TEXT_COLOR for readability
                text = font.render(annotation.label, True, ANNOTATION_TEXT_COLOR) 
                text_x = x1 + (x2 - x1) // 2 - text.get_width() // 2
                text_y = rect.centery - text.get_height() // 2
                
                # Draw the text directly (no background rectangle needed with good contrast)
                surface.blit(text, (text_x, text_y))
            else:
                # Draw tiny indicator (maybe less necessary now, but keep for very small segments)
                # Ensure indicator color contrasts with background
                indicator_color = ANNOTATION_TEXT_COLOR # Use white indicator
                pygame.draw.rect(surface, indicator_color, (x1, rect.top + 5, max(1, x2 - x1), 3), 0)
    
    def handle_event(self, event: pygame.event.Event, time_scale: Optional[TimeScale]) -> bool:
        """Handle mouse events. Returns True if event was handled."""
        if not self.visible:
            return False
            
        if event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
            self.highlight = self.rect.collidepoint(mouse_x, mouse_y)
            return False
            
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            mouse_x, mouse_y = event.pos
            
            # Check for collapse/expand button
            button_rect = pygame.Rect(self.rect.right - 25, self.rect.top + 5, 20, 20)
            if button_rect.collidepoint(mouse_x, mouse_y):
                self.toggle_collapsed()
                return True
                
            # Handle click in header (select channel)
            header_rect = pygame.Rect(self.rect.left, self.rect.top, self.rect.width, 30)
            if header_rect.collidepoint(mouse_x, mouse_y):
                self.selected = not self.selected
                return True
            
            # Handle click on annotations in content area
            if self.annotation_channel and time_scale and not self.collapsed:
                content_rect = pygame.Rect(
                    self.rect.left + 1, self.rect.top + 31,
                    self.rect.width - 2, self.rect.height - 32
                )
                
                if content_rect.collidepoint(mouse_x, mouse_y):
                    # Calculate time at click position
                    click_unit = (mouse_x - content_rect.left) / content_rect.width
                    click_time = time_scale.to_scale(click_unit)
                    
                    # Find annotation at this time if any
                    clicked_annotation = self.annotation_channel.get_annotation_at_time(click_time)
                    
                    # If double click and annotation found, edit it
                    if hasattr(event, 'double') and event.double and clicked_annotation:
                        self.edit_annotation = clicked_annotation
                        return True
                    
                    # Single click on annotation selects this channel
                    if clicked_annotation:
                        self.selected = True
                        return True
                
        return False


class AutocompleteSearch:
    """Class to handle efficient search and autocomplete for large label sets."""
    
    def __init__(self, items: List[str]):
        self.items = items
        self.current_query = ""
        self.current_results = []
        self.selection_index = 0
        
        # Create normalized versions for case insensitive search
        self.normalized_items = [item.lower() for item in items]
    
    def update_items(self, items: List[str]):
        """Update the list of searchable items."""
        self.items = items
        self.normalized_items = [item.lower() for item in items]
        self.reset()
    
    def reset(self):
        """Reset search state."""
        self.current_query = ""
        self.current_results = []
        self.selection_index = 0
    
    def update_query(self, query: str) -> List[str]:
        """Update search query and return matching results."""
        self.current_query = query.lower()
        
        if not query:
            self.current_results = []
        else:
            # Create list of (index, score) pairs for ranking
            matches = []
            query_lower = query.lower()
            
            for idx, (item, item_lower) in enumerate(zip(self.items, self.normalized_items)):
                # Skip if no match at all
                if query_lower not in item_lower:
                    continue
                    
                # Compute ranking score (smaller is better)
                # 0: exact match
                # 1: prefix match
                # 2: contains match
                # Longer strings have slightly higher score within category
                if item_lower == query_lower:
                    score = 0
                elif item_lower.startswith(query_lower):
                    score = 1 + 0.001 * len(item)
                else:
                    score = 2 + 0.001 * len(item)
                
                matches.append((idx, score))
            
            # Sort by score and get original items
            matches.sort(key=lambda x: x[1])
            self.current_results = [self.items[idx] for idx, _ in matches]
            
        self.selection_index = 0 if self.current_results else -1
        return self.current_results
    
    def navigate(self, direction: int) -> None:
        """Navigate through results (1 for down, -1 for up)."""
        if not self.current_results:
            return
            
        self.selection_index = (self.selection_index + direction) % len(self.current_results)
    
    def get_selected(self) -> Optional[str]:
        """Get the currently selected item."""
        if not self.current_results or self.selection_index < 0:
            return None
        return self.current_results[self.selection_index]
    
    def get_results(self) -> List[str]:
        """Get current results."""
        return self.current_results


class Logger:
    """Simple logger for the annotation tool."""
    
    def __init__(self, max_messages: int = 100):
        self.messages = []
        self.max_messages = max_messages
        self.listeners = []
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.messages.append(log_entry)
        
        # Trim if necessary
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Notify listeners
        for listener in self.listeners:
            listener(log_entry, level)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, "ERROR")
    
    def success(self, message: str) -> None:
        """Log a success message."""
        self.log(message, "SUCCESS")
        
    def add_listener(self, listener: Callable[[str, str], None]) -> None:
        """Add a listener function to be called for new log entries."""
        self.listeners.append(listener)
    
    def get_recent(self, count: int = 10) -> List[str]:
        """Get the most recent log entries."""
        return self.messages[-count:]


class TimeScrubber:
    """Widget to navigate through time."""
    
    def __init__(self, time_scale: Optional[TimeScale], rect: pygame.Rect):
        self.time_scale = time_scale
        self.rect = rect
        self.cursor_pos = 0.5  # Cursor position in [0,1]
    
    def set_position(self, pos: float) -> None:
        """Set cursor position in normalized [0,1] coordinates."""
        self.cursor_pos = max(0, min(1, pos))
    
    def get_current_time(self) -> Optional[np.datetime64]:
        """Get the current time at cursor position."""
        if self.time_scale is None:
            return None
        return self.time_scale.to_scale(self.cursor_pos)
    
    def set_current_time(self, time: np.datetime64) -> None:
        """Set cursor position to the given time."""
        if self.time_scale is None:
            return
        units = self.time_scale.to_unit(np.array([time]))[0]
        self.cursor_pos = max(0, min(1, units))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle keyboard events. Returns True if event was handled."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 0.01)
                return True
            elif event.key == pygame.K_RIGHT:
                self.cursor_pos = min(1, self.cursor_pos + 0.01)
                return True
        return False
    
    def render(self, surface: pygame.Surface) -> None:
        """Render the time scrubber."""
        # If no time scale, just draw a placeholder
        if self.time_scale is None:
            font = pygame.font.SysFont("Helvetica Neue", 14)
            text = font.render("No time data loaded", True, TEXT_COLOR)
            surface.blit(text, (self.rect.centerx - text.get_width() // 2, 
                               self.rect.centery - text.get_height() // 2))
            return
        
        # Draw time range labels
        font = pygame.font.SysFont("Helvetica Neue", 14)
        
        # Get time strings
        min_time_str = self.time_scale.time_to_str(self.time_scale.min_time)
        max_time_str = self.time_scale.time_to_str(self.time_scale.max_time)
        
        # Create text surfaces
        min_label = font.render(min_time_str, True, TEXT_COLOR)
        max_label = font.render(max_time_str, True, TEXT_COLOR)
        
        # Draw the start time and end time on a single line at the bottom
        timestamp_y = self.rect.bottom - min_label.get_height() - 5  
        
        # Draw start time at left edge
        left_margin = 5
        surface.blit(min_label, (self.rect.left + left_margin, timestamp_y))
        
        # Draw end time at right edge
        right_margin = 5
        max_label_x = self.rect.right - max_label.get_width() - right_margin
        surface.blit(max_label, (max_label_x, timestamp_y))
        
        # Remove cursor line and current time rendering from TimeScrubber
        # This will be handled by the main AnnotationTool rendering loop


class Channel:
    """Base class for all channel types."""
    
    def __init__(self, name: str, rect: pygame.Rect):
        self.name = name
        self.rect = rect
        self.selected = False
        self.visible = True
        self.collapsed = False
        self.data_source = None
        self.annotation_channel = None
        self.header_height = 30  # Fixed header height
        self.min_content_height = 100  # Minimum height for data channels
        self.min_annotation_height = 40  # Minimum height for annotation channels
    
    def get_height(self) -> int:
        """Get the current height of the channel based on its state."""
        if self.collapsed:
            return self.header_height  # Only show header when collapsed
        elif isinstance(self.data_source, ImageDataSource):
            # Use the data source's get_height method
            return self.data_source.get_height()
        elif self.annotation_channel:
            return max(self.min_annotation_height, 100)  # Ensure minimum height for annotation channels
        else:
            return self.min_content_height
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events. Returns True if event was handled."""
        if not self.visible:
            return False
            
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            mouse_x, mouse_y = event.pos
            
            # Check for collapse/expand button
            button_rect = pygame.Rect(self.rect.right - 25, self.rect.top + 5, 20, 20)
            if button_rect.collidepoint(mouse_x, mouse_y):
                self.collapsed = not self.collapsed
                return True
            
            # Check for y+ button (only for scalar/vector data sources)
            if isinstance(self.data_source, (ScalarDataSource, VectorDataSource)):
                y_plus_rect = pygame.Rect(self.rect.right - 50, self.rect.top + 5, 20, 20)
                if y_plus_rect.collidepoint(mouse_x, mouse_y):
                    self.data_source.scale_y(1.2)  # Zoom in by 20%
                    return True
                
                # Check for y reset button
                y_reset_rect = pygame.Rect(self.rect.right - 75, self.rect.top + 5, 20, 20)
                if y_reset_rect.collidepoint(mouse_x, mouse_y):
                    self.data_source.y_scale = 1.0  # Reset to default scale
                    return True
                
                # Check for y- button
                y_minus_rect = pygame.Rect(self.rect.right - 100, self.rect.top + 5, 20, 20)
                if y_minus_rect.collidepoint(mouse_x, mouse_y):
                    self.data_source.scale_y(0.8)  # Zoom out by 20%
                    return True
            
            # Check for display mode toggle button (only for ImageDataSource)
            if isinstance(self.data_source, ImageDataSource) and self.data_source.max_images_in_view > 1:
                mode_button_rect = pygame.Rect(self.rect.right - 50, self.rect.top + 5, 20, 20)
                if mode_button_rect.collidepoint(mouse_x, mouse_y):
                    # Only toggle mode if we have more than one image in view
                    if len(self.data_source.times) > 1:
                        # Toggle between grid and centered modes
                        self.data_source.display_mode = "centered" if self.data_source.display_mode == "grid" else "grid"
                    return True
                
            # Handle click in header (select channel)
            header_rect = pygame.Rect(self.rect.left, self.rect.top, self.rect.width, self.header_height)
            if header_rect.collidepoint(mouse_x, mouse_y):
                self.selected = not self.selected
                return True
        
        return False
    
    def render(self, surface: pygame.Surface, time_scale: Optional[TimeScale] = None) -> None:
        """Render the channel."""
        if not self.visible:
            return
        
        # Draw channel background
        if self.selected:
            pygame.draw.rect(surface, SELECTED_CHANNEL_COLOR, self.rect)
        else:
            pygame.draw.rect(surface, GRID_COLOR, self.rect, 1)
        
        # Draw channel name
        font = pygame.font.SysFont("Helvetica Neue", 14) # Reduced size
        text = font.render(self.name, True, TEXT_COLOR)
        surface.blit(text, (self.rect.left + 5, self.rect.top + 5))
        
        # Draw collapse/expand button
        button_rect = pygame.Rect(self.rect.right - 25, self.rect.top + 5, 20, 20)
        pygame.draw.rect(surface, GRID_COLOR, button_rect, 1)
        
        if self.collapsed:
            # Draw plus sign
            pygame.draw.line(surface, TEXT_COLOR, 
                           (button_rect.centerx - 5, button_rect.centery),
                           (button_rect.centerx + 5, button_rect.centery), 1)
            pygame.draw.line(surface, TEXT_COLOR,
                           (button_rect.centerx, button_rect.centery - 5),
                           (button_rect.centerx, button_rect.centery + 5), 1)
        else:
            # Draw minus sign
            pygame.draw.line(surface, TEXT_COLOR,
                           (button_rect.centerx - 5, button_rect.centery),
                           (button_rect.centerx + 5, button_rect.centery), 1)
        
        # Draw y+ and y- buttons for scalar/vector data sources
        if isinstance(self.data_source, (ScalarDataSource, VectorDataSource)):
            # Draw y+ button
            y_plus_rect = pygame.Rect(self.rect.right - 50, self.rect.top + 5, 20, 20)
            pygame.draw.rect(surface, GRID_COLOR, y_plus_rect, 1)
            # Draw up arrow
            pygame.draw.line(surface, TEXT_COLOR,
                           (y_plus_rect.centerx - 5, y_plus_rect.centery + 3),
                           (y_plus_rect.centerx, y_plus_rect.centery - 3), 1)
            pygame.draw.line(surface, TEXT_COLOR,
                           (y_plus_rect.centerx + 5, y_plus_rect.centery + 3),
                           (y_plus_rect.centerx, y_plus_rect.centery - 3), 1)
            
            # Draw y reset button
            y_reset_rect = pygame.Rect(self.rect.right - 75, self.rect.top + 5, 20, 20)
            pygame.draw.rect(surface, GRID_COLOR, y_reset_rect, 1)
            # Draw "Y" text
            y_font = pygame.font.SysFont("Helvetica Neue", 12)
            y_text = y_font.render("Y", True, TEXT_COLOR)
            surface.blit(y_text, (
                y_reset_rect.centerx - y_text.get_width() // 2,
                y_reset_rect.centery - y_text.get_height() // 2
            ))
            
            # Draw y- button
            y_minus_rect = pygame.Rect(self.rect.right - 100, self.rect.top + 5, 20, 20)
            pygame.draw.rect(surface, GRID_COLOR, y_minus_rect, 1)
            # Draw down arrow
            pygame.draw.line(surface, TEXT_COLOR,
                           (y_minus_rect.centerx - 5, y_minus_rect.centery - 3),
                           (y_minus_rect.centerx, y_minus_rect.centery + 3), 1)
            pygame.draw.line(surface, TEXT_COLOR,
                           (y_minus_rect.centerx + 5, y_minus_rect.centery - 3),
                           (y_minus_rect.centerx, y_minus_rect.centery + 3), 1)
        
        # Draw display mode toggle button for thumbnail channels
        if isinstance(self.data_source, ImageDataSource) and self.data_source.max_images_in_view > 1:
            mode_button_rect = pygame.Rect(self.rect.right - 50, self.rect.top + 5, 20, 20)
            pygame.draw.rect(surface, GRID_COLOR, mode_button_rect, 1)
            
            # Draw mode indicator (G for grid, C for centered)
            mode_text = "G" if self.data_source.display_mode == "grid" else "C"
            mode_font = pygame.font.SysFont("Helvetica Neue", 12)
            mode_surface = mode_font.render(mode_text, True, TEXT_COLOR)
            surface.blit(mode_surface, (
                mode_button_rect.centerx - mode_surface.get_width() // 2,
                mode_button_rect.centery - mode_surface.get_height() // 2
            ))
        
        # Draw content if not collapsed
        if not self.collapsed:
            # Create content rectangle (below the header)
            content_rect = pygame.Rect(
                self.rect.left + 1,
                self.rect.top + self.header_height,  # Start after header
                self.rect.width - 2,
                self.rect.height - self.header_height - 1  # Use remaining height
            )
            
            if self.data_source:
                self.data_source.render(surface, time_scale, content_rect)
            elif self.annotation_channel:
                self.annotation_channel.render(surface, time_scale, content_rect)

class ChannelView:
    """Widget to display and manage channels."""
    
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.channels: List[Channel] = []
        self.selected_channel: Optional[Channel] = None
        self.scroll_offset = 0
        self.min_channel_height = 150  # Default height for data channels
        self.min_annotation_height = 70  # Height for annotation channels (header + content)
        self.padding = 5  # Vertical padding between channels
        self.left_margin = 5  # Left margin for labels
        self.right_margin = 5  # Right margin for values
        self.scrollbar_width = 10
        self.is_dragging_scrollbar = False
        self.drag_start_y = 0
        self.drag_start_offset = 0
    
    def get_channel_height(self, channel: Channel) -> int:
        """Get the appropriate height for a channel based on its type and state."""
        return channel.get_height()
    
    def get_channel_position(self, index: int) -> int:
        """Calculate the Y position of a channel given its index."""
        y = 0
        for i in range(index):
            y += self.get_channel_height(self.channels[i]) + self.padding
        return y
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle keyboard and mouse events. Returns True if event was handled."""
        # First check if any channel handles the event
        for channel in self.channels:
            if channel.handle_event(event):
                return True
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if self.selected_channel:
                    idx = self.channels.index(self.selected_channel)
                    if idx > 0:
                        self.selected_channel = self.channels[idx - 1]
                        # Adjust scroll to show selected channel
                        channel_y = self.get_channel_position(idx - 1)
                        if channel_y < self.scroll_offset:
                            self.scroll_offset = channel_y
                return True
            elif event.key == pygame.K_DOWN:
                if self.selected_channel:
                    idx = self.channels.index(self.selected_channel)
                    if idx < len(self.channels) - 1:
                        self.selected_channel = self.channels[idx + 1]
                        # Adjust scroll to show selected channel
                        next_channel_y = self.get_channel_position(idx + 1)
                        next_channel_height = self.get_channel_height(self.channels[idx + 1])
                        if next_channel_y + next_channel_height > self.scroll_offset + self.rect.height:
                            self.scroll_offset = next_channel_y + next_channel_height - self.rect.height
                return True
            elif event.key == pygame.K_PAGEUP:
                self.scroll_offset = max(0, self.scroll_offset - self.rect.height)
                return True
            elif event.key == pygame.K_PAGEDOWN:
                total_height = sum(self.get_channel_height(ch) for ch in self.channels)
                total_height += (len(self.channels) - 1) * self.padding
                max_scroll = max(0, total_height - self.rect.height)
                self.scroll_offset = min(max_scroll, self.scroll_offset + self.rect.height)
                return True
        
        # Handle mouse wheel scrolling
        elif event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                # Calculate total content height
                total_height = sum(self.get_channel_height(ch) for ch in self.channels)
                total_height += (len(self.channels) - 1) * self.padding
                
                # Calculate scroll amount (negative because wheel up should scroll up)
                scroll_amount = -event.y * 30
                
                # Update scroll offset
                self.scroll_offset = max(0, min(
                    self.scroll_offset + scroll_amount,
                    max(0, total_height - self.rect.height)
                ))
                return True
        
        # Handle scrollbar dragging
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos
            if (self.rect.right - self.scrollbar_width <= mouse_x <= self.rect.right and
                self.rect.top <= mouse_y <= self.rect.bottom):
                # Calculate total content height
                total_height = sum(self.get_channel_height(ch) for ch in self.channels)
                total_height += (len(self.channels) - 1) * self.padding
                
                if total_height > self.rect.height:
                    self.is_dragging_scrollbar = True
                    self.drag_start_y = mouse_y
                    self.drag_start_offset = self.scroll_offset
                    return True
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging_scrollbar = False
        
        elif event.type == pygame.MOUSEMOTION and self.is_dragging_scrollbar:
            mouse_y = event.pos[1]
            # Calculate total content height
            total_height = sum(self.get_channel_height(ch) for ch in self.channels)
            total_height += (len(self.channels) - 1) * self.padding
            
            if total_height > self.rect.height:
                # Calculate scroll ratio
                scroll_ratio = (mouse_y - self.drag_start_y) / self.rect.height
                # Update scroll offset
                self.scroll_offset = max(0, min(
                    self.drag_start_offset + scroll_ratio * total_height,
                    total_height - self.rect.height
                ))
                return True
        
        return False
    
    def render(self, surface: pygame.Surface, time_scale: Optional[TimeScale] = None) -> None:
        """Render the channel view."""
        # Draw background for entire view
        pygame.draw.rect(surface, BACKGROUND_COLOR, self.rect)
        
        # Calculate visible area and total height
        visible_height = self.rect.height
        total_height = sum(self.get_channel_height(ch) for ch in self.channels)
        total_height += (len(self.channels) - 1) * self.padding  # Add padding between channels
        
        # Calculate maximum scroll offset
        max_scroll = max(0, total_height - visible_height)
        self.scroll_offset = min(self.scroll_offset, max_scroll)
        
        # Calculate content width (excluding margins)
        content_width = self.rect.width - (self.left_margin + self.right_margin)
        
        # Draw channels
        y = self.rect.top - self.scroll_offset
        for channel in self.channels:
            # Get appropriate height for this channel
            channel_height = self.get_channel_height(channel)
            
            # Create channel rectangle with margins
            channel_rect = pygame.Rect(
                self.rect.left + self.left_margin,  # Add left margin
                y,
                content_width,  # Use content width (excluding margins)
                channel_height
            )
            
            # Only render if channel is visible
            if (y + channel_height > self.rect.top and 
                y < self.rect.bottom):
                channel.rect = channel_rect
                channel.render(surface, time_scale)
            
            # Move to next channel position
            y += channel_height + self.padding
            
        # Draw scrollbar if content is scrollable
        if total_height > visible_height:
            # Calculate scrollbar dimensions
            scrollbar_height = (visible_height / total_height) * visible_height
            scrollbar_x = self.rect.right - self.scrollbar_width
            scrollbar_y = self.rect.top + (self.scroll_offset / total_height) * visible_height
            
            # Draw scrollbar track
            pygame.draw.rect(surface, GRID_COLOR, 
                           (scrollbar_x, self.rect.top, self.scrollbar_width, visible_height))
            
            # Draw scrollbar thumb
            pygame.draw.rect(surface, HIGHLIGHT_COLOR,
                           (scrollbar_x, scrollbar_y, self.scrollbar_width, scrollbar_height))

    def add_channel(self, channel: Channel) -> None:
        """Add a channel to the view."""
        self.channels.append(channel)
        # If this is the first channel and it's an annotation channel, select it
        if len(self.channels) == 1 and channel.annotation_channel:
            self.selected_channel = channel
            channel.selected = True


class LabelEditor:
    """Widget for editing annotation labels."""
    

class LabelEditor:
    """Widget for editing annotation labels."""
    
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.text = ""
        self.active = False
        self.font = pygame.font.SysFont("Helvetica Neue", 20) # Reduced size
        self.cursor_pos = 0
        self.cursor_visible = True
        self.cursor_blink_time = 0
        self.cursor_blink_rate = 500  # milliseconds
        self.annotation_channel = None
        self.time_scale = None
        self.autocomplete = None
        self.suggestion_rect = pygame.Rect(rect.left, rect.bottom + 5, rect.width, 150)
        self.max_suggestions = 5
        self.on_submit = None  # Callback for when a label is submitted
        
    def activate(self, annotation_channel, on_submit):
        """Activate the editor with the given annotation channel."""
        self.active = True
        self.annotation_channel = annotation_channel
        self.text = ""
        self.cursor_pos = 0
        self.on_submit = on_submit
        # Initialize autocomplete with possible labels
        self.autocomplete = AutocompleteSearch(annotation_channel.possible_labels)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle keyboard events. Returns True if event was handled."""
        if not self.active:
            return False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.autocomplete and self.autocomplete.get_selected():
                    # Use the selected suggestion
                    self.text = self.autocomplete.get_selected()
                # Submit the label
                if self.text and self.on_submit:
                    self.on_submit(self.text)
                self.active = False
                return True
            elif event.key == pygame.K_ESCAPE:
                self.active = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos - 1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
                    if self.autocomplete:
                        self.autocomplete.update_query(self.text)
                return True
            elif event.key == pygame.K_LEFT:
                if self.cursor_pos > 0:
                    self.cursor_pos -= 1
                return True
            elif event.key == pygame.K_RIGHT:
                if self.cursor_pos < len(self.text):
                    self.cursor_pos += 1
                return True
            elif event.key == pygame.K_UP and self.autocomplete:
                self.autocomplete.navigate(-1)
                return True
            elif event.key == pygame.K_DOWN and self.autocomplete:
                self.autocomplete.navigate(1)
                return True
            elif event.key == pygame.K_TAB and self.autocomplete:
                # Complete with the selected suggestion
                selected = self.autocomplete.get_selected()
                if selected:
                    self.text = selected
                    self.cursor_pos = len(self.text)
                return True
            elif event.unicode:
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
                if self.autocomplete:
                    self.autocomplete.update_query(self.text)
                return True
        return False
    
    def render(self, surface: pygame.Surface) -> None:
        """Render the label editor."""
        if not self.active:
            return
            
        # Draw background
        pygame.draw.rect(surface, GRID_COLOR, self.rect)
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, 1)
        
        # Draw text
        text_surface = self.font.render(self.text, True, TEXT_COLOR)
        surface.blit(text_surface, (self.rect.left + 5, self.rect.centery - text_surface.get_height() // 2))
        
        # Draw cursor
        if self.cursor_visible:
            cursor_x = self.rect.left + 5
            if self.cursor_pos > 0:
                cursor_x += self.font.size(self.text[:self.cursor_pos])[0]
            cursor_y = self.rect.centery
            pygame.draw.line(surface, TEXT_COLOR,
                           (cursor_x, cursor_y - 10),
                           (cursor_x, cursor_y + 10), 1)
        
        # Draw suggestions if we have any
        if self.autocomplete:
            suggestions = self.autocomplete.get_results()[:self.max_suggestions]
            if suggestions:
                # Draw suggestion background
                suggestion_height = len(suggestions) * 25 + 10
                suggestion_rect = pygame.Rect(
                    self.suggestion_rect.left,
                    self.suggestion_rect.top,
                    self.suggestion_rect.width,
                    suggestion_height
                )
                pygame.draw.rect(surface, BACKGROUND_COLOR, suggestion_rect)
                pygame.draw.rect(surface, GRID_COLOR, suggestion_rect, 1)
                
                # Draw each suggestion
                for i, suggestion in enumerate(suggestions):
                    selected = (self.autocomplete.selection_index == i)
                    color = HIGHLIGHT_COLOR if selected else TEXT_COLOR
                    text_surface = self.font.render(suggestion, True, color)
                    surface.blit(text_surface, (
                        suggestion_rect.left + 5,
                        suggestion_rect.top + 5 + i * 25
                    ))
    
    def update(self, dt: int) -> None:
        """Update cursor blink state."""
        if not self.active:
            return
            
        self.cursor_blink_time += dt
        if self.cursor_blink_time >= self.cursor_blink_rate:
            self.cursor_visible = not self.cursor_visible
            self.cursor_blink_time = 0


class StatusBar:
    """Status bar widget showing current state and tools."""
    
    def __init__(self, rect: pygame.Rect, logger: Logger):
        self.rect = rect
        self.logger = logger
        self.status_text = ""
        self.status_color = TEXT_COLOR
        self.left_buttons = []
        self.right_buttons = []
        self.channel_info = ""  # Store current channel info
        
        # Add basic buttons to the left side
        self.add_button("Save", self._dummy_callback)
        
        # Add buttons to the right side
        self.add_button("Help", self._dummy_callback)
        
        # Listen for log messages
        logger.add_listener(self._on_log_message)
    
    def _dummy_callback(self) -> None:
        """Placeholder callback."""
        pass
    
    def _on_log_message(self, message: str, level: str) -> None:
        """Update status text when a log message arrives."""
        self.status_text = message.split("] ", 2)[-1]  # Remove timestamp and level
        
        if level == "ERROR":
            self.status_color = ERROR_COLOR
        elif level == "WARNING":
            self.status_color = WARNING_COLOR
        elif level == "SUCCESS":
            self.status_color = SUCCESS_COLOR
        else:
            self.status_color = TEXT_COLOR
    
    def set_channel_info(self, info: str) -> None:
        """Update the current channel information."""
        self.channel_info = info
    
    def add_button(self, text: str, callback: Callable[[], None]) -> None:
        """Add a button to the status bar (stores text and callback only)."""
        button_data = {
            "text": text,
            "callback": callback
            # Rect will be calculated dynamically in render
        }
        
        # For now, assume Save/Help are added first and are the only left buttons
        # If more buttons are needed, this logic might need adjustment
        if text == "Save" or text == "Help":
             self.left_buttons.append(button_data)
        # else: # Example if adding right buttons later
        #    self.right_buttons.append(button_data)
    
    def render(self, surface: pygame.Surface) -> None:
        """Render the status bar."""
        # Draw background
        pygame.draw.rect(surface, BACKGROUND_COLOR, self.rect)
        pygame.draw.line(surface, GRID_COLOR, (self.rect.left, self.rect.top), 
                        (self.rect.right, self.rect.top), 1)
        
        # --- Draw Buttons Dynamically --- 
        font = pygame.font.SysFont("Helvetica Neue", 14) # Reduced size
        button_padding = 10
        current_x = self.rect.left + button_padding
        button_height = 24
        button_y = self.rect.centery - button_height // 2
        
        for button_data in self.left_buttons: 
            text = button_data["text"]
            text_surface = font.render(text, True, TEXT_COLOR)
            button_width = text_surface.get_width() + 16 # Add horizontal padding
            
            # Calculate button rect dynamically
            button_rect = pygame.Rect(current_x, button_y, button_width, button_height)
            
            # Draw button background and border
            pygame.draw.rect(surface, BACKGROUND_COLOR, button_rect) # Background fill
            pygame.draw.rect(surface, GRID_COLOR, button_rect, 1) # Border
            
            # Draw button text centered
            surface.blit(text_surface, (button_rect.centerx - text_surface.get_width() // 2, 
                                       button_rect.centery - text_surface.get_height() // 2))
            
            # Store the calculated rect in the button data for potential click handling
            button_data["rect"] = button_rect 
                                       
            # Update current_x for the next button
            current_x = button_rect.right + button_padding

        # --- Status Text --- 
        status_text_x = current_x # Position status text after the last button
        if self.status_text:
            status_font = pygame.font.SysFont("Helvetica Neue", 14) # Reduced size
            text = status_font.render(self.status_text, True, self.status_color)
            surface.blit(text, (status_text_x, self.rect.centery - text.get_height() // 2))
        
        # --- Channel Info --- 
        # Draw channel info on the right (add logic for right_buttons if needed)
        if self.channel_info:
            info_font = pygame.font.SysFont("Helvetica Neue", 14) # Changed from 16 to 14 to match buttons
            text = info_font.render(self.channel_info, True, TEXT_COLOR)
            # Position relative to the right edge
            info_x = self.rect.right - text.get_width() - button_padding
            surface.blit(text, (info_x, self.rect.centery - text.get_height() // 2))


class AnnotationTool:
    """Main application class for the annotation tool."""
    
    def __init__(self, width: int = None, height: int = None, annotation_dir: str = "annotations", load_existing: bool = False):
        """Initialize the annotation tool."""
        pygame.init()
        
        # Get display info to determine a large window size
        display_info = pygame.display.Info()
        # Use 90% of screen size for a large, non-fullscreen window
        self.screen_width = int(display_info.current_w * 0.9)
        self.screen_height = int(display_info.current_h * 0.9)
        
        # Create a large, resizable window (not fullscreen)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Time Series Annotation Tool")
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.annotation_dir = annotation_dir
        
        # Initialize components
        self.logger = Logger()
        self.time_scale = None
        
        # Initialize layout-related variables (will be set in _update_layout)
        self.status_rect = None
        self.scrubber_rect = None
        self.channel_rect = None
        self.label_rect = None
        
        # Initial layout calculation
        self._update_layout(self.screen_width, self.screen_height)
        
        # Set up components using the calculated rects
        self.status_bar = StatusBar(self.status_rect, self.logger)
        self.time_scrubber = TimeScrubber(self.time_scale, self.scrubber_rect)
        self.channel_view = ChannelView(self.channel_rect)
        self.label_editor = LabelEditor(self.label_rect)
        
        # Update status bar buttons with real callbacks
        self.status_bar.left_buttons = []  # Clear default buttons
        self.status_bar.right_buttons = []
        self.status_bar.add_button("Save", self._handle_save)
        self.status_bar.add_button("Help", self._handle_help)
        
        # Set up keyboard shortcuts
        self.shortcuts = {
            pygame.K_ESCAPE: self._handle_escape,
            pygame.K_SPACE: self._handle_space,
            pygame.K_DELETE: self._handle_delete,
            pygame.K_BACKSPACE: self._handle_delete,  # Add backspace as delete
            pygame.K_EQUALS: self._handle_zoom_in,
            pygame.K_MINUS: self._handle_zoom_out,
            pygame.K_LEFT: self._handle_left,
            pygame.K_RIGHT: self._handle_right,
            pygame.K_UP: self._handle_up,
            pygame.K_DOWN: self._handle_down,
            pygame.K_TAB: self._handle_tab,
            pygame.K_F1: self._handle_help,
            pygame.K_RETURN: self._handle_enter,
            pygame.K_l: self._handle_l,
            pygame.K_r: self._handle_r,
            pygame.K_c: self._handle_c,  # Add 'c' key for changing labels
        }
        
        # Set up state
        self.selected_channel_index = 0
        self.selected_annotation_channel_index = -1 # Track the index within annotation_channels list
        self.show_help = False
        self.is_fullscreen = True  # Start in fullscreen mode
        
        # Annotation state
        self.creating_annotation = False
        self.annotation_start_time = None
        self.annotation_channel = None
        self.annotation_label = ""
        
        # Add new state variables for annotation editing
        self.editing_annotation = None  # Currently selected annotation for editing
        self.editing_edge = None  # 'left' or 'right' edge being edited
        
        # Set up colors
        self.colors = {
            "background": BACKGROUND_COLOR,
            "grid": GRID_COLOR,
            "text": TEXT_COLOR,
            "highlight": HIGHLIGHT_COLOR,
            "error": ERROR_COLOR,
            "warning": WARNING_COLOR,
            "success": SUCCESS_COLOR
        }
        
        # Setup cursor blink timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 500)  # 500ms blink rate
        
        # Add time indicator alpha
        self.time_indicator_alpha = 128  # Semi-transparent
        
        # Create annotation directory if it doesn't exist
        if not os.path.exists(self.annotation_dir):
            os.makedirs(self.annotation_dir)
            self.logger.info(f"Created annotation directory: {self.annotation_dir}")
            
        # Add key state tracking for continuous scrolling
        self.key_states = {
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False
        }
        self.scroll_speed = 0.0001  # Very slow initial speed
        self.scroll_acceleration = 0.00006  # Gentle acceleration
        self.current_scroll_speed = self.scroll_speed  # Current scroll speed
        self.max_scroll_speed = 0.15  # Maximum scroll speed
        self.last_scroll_time = pygame.time.get_ticks()  # For timing scroll updates
        self.key_press_time = {}  # Track when each key was pressed
    
    def _update_layout(self, width, height):
        """Recalculate UI component positions and sizes based on window dimensions."""
        self.screen_width = width
        self.screen_height = height
        
        # Recalculate heights and total bottom height (could be adjusted based on width/height if desired)
        self.status_height = 30
        self.timestamp_height = 20
        self.scrubber_height = 30
        self.total_bottom_height = self.status_height + self.timestamp_height + self.scrubber_height

        # Recalculate Rects
        self.status_rect = pygame.Rect(0, self.screen_height - self.status_height, self.screen_width, self.status_height)
        self.scrubber_rect = pygame.Rect(
            0, 
            self.screen_height - self.total_bottom_height, 
            self.screen_width, 
            self.timestamp_height + self.scrubber_height
        )
        self.channel_rect = pygame.Rect(0, 0, self.screen_width, self.screen_height - self.total_bottom_height)
        self.label_rect = pygame.Rect(self.screen_width // 4, self.screen_height // 2 - 15, self.screen_width // 2, 30)
        
        # Update component rects if they already exist
        if hasattr(self, 'status_bar') and self.status_bar:
            self.status_bar.rect = self.status_rect
        if hasattr(self, 'time_scrubber') and self.time_scrubber:
            self.time_scrubber.rect = self.scrubber_rect
        if hasattr(self, 'channel_view') and self.channel_view:
            self.channel_view.rect = self.channel_rect
        if hasattr(self, 'label_editor') and self.label_editor:
            self.label_editor.rect = self.label_rect
            # Also update suggestion rect position if editor is active
            self.label_editor.suggestion_rect = pygame.Rect(self.label_rect.left, self.label_rect.bottom + 5, self.label_rect.width, 150)
    
    def _on_annotation_save(self, annotation: Annotation):
        """Callback when an annotation is saved through the label editor."""
        if self.annotation_channel:
            # Create the annotation with the current time as the end time
            current_time = self.time_scrubber.get_current_time()
            annotation = Annotation(
                label=self.label_editor.text,
                start_time=self.annotation_start_time,
                end_time=current_time
            )
            
            # Add to channel
            self.annotation_channel.add_annotation(annotation)
            
            # Reset state
            self.creating_annotation = False
            self.annotation_start_time = None
            self.annotation_channel = None
            self.label_editor.active = False
            
            # Log success
            self.logger.success(f"Updated annotation: {annotation.label} ({self.time_scale.time_to_str(annotation.start_time)} - {self.time_scale.time_to_str(annotation.end_time)})")
    
    def load_data(self, data_sources: List[DataSource], annotation_channels: List[AnnotationChannel], load_existing: bool = False):
        """Load data sources and annotation channels."""
        if not data_sources and not annotation_channels:
            self.logger.error("No data sources or annotation channels provided")
            return
            
        # Find overall time range for data bounds
        min_time = None
        max_time = None
        
        for source in data_sources:
            source_min, source_max = source.get_time_range()
            if min_time is None or source_min < min_time:
                min_time = source_min
            if max_time is None or source_max > max_time:
                max_time = source_max
        
        if min_time is None or max_time is None:
            min_time = np.datetime64('2000-01-01')
            max_time = min_time + np.timedelta64(1, 'h')
        
        # Initialize time scale with full range first
        self.time_scale = TimeScale(min_time, max_time)
        self.time_scrubber.time_scale = self.time_scale
        
        # Clear existing channels
        self.channel_view.channels = []
        
        # Add data source channels in their original order
        for source in data_sources:
            rect = pygame.Rect(0, 0, self.screen.get_width(), 150)
            channel = Channel(source.name, rect)
            channel.data_source = source
            source.time_scale = self.time_scale
            self.channel_view.add_channel(channel)
        
        # Add annotation channels
        for ann_channel in annotation_channels:
            rect = pygame.Rect(0, 0, self.screen.get_width(), 150)
            channel = Channel(ann_channel.name, rect)
            channel.annotation_channel = ann_channel
            self.channel_view.add_channel(channel)
        
        # Load existing annotations if requested
        if load_existing:
            self.load_annotations()  # Load all channels
        
        # Now that annotations are loaded, find the latest time to position cursor
        latest_time = None
        
        # First check for annotations
        for channel in self.channel_view.channels:
            if channel.annotation_channel and channel.annotation_channel.annotations:
                # Get the latest end time from annotations
                channel_latest = max(a.end_time for a in channel.annotation_channel.annotations)
                if latest_time is None or channel_latest > latest_time:
                    latest_time = channel_latest
        
        # If no annotations found, use the latest start time from data sources
        if latest_time is None:
            for source in data_sources:
                source_min, _ = source.get_time_range()
                if latest_time is None or source_min > latest_time:
                    latest_time = source_min
        
        # Set cursor position to the latest time found
        if latest_time is not None:
            # First set the cursor position
            unit_pos = self.time_scale.to_unit(np.array([latest_time]))[0]
            self.time_scrubber.set_position(unit_pos)
            
            # Then adjust the view window to be centered on this position
            view_duration = np.timedelta64(10, 'm')  # 10 minutes total (5 on each side)
            view_start = latest_time - view_duration / 2
            view_end = latest_time + view_duration / 2
            
            # Ensure view stays within data bounds
            view_start = max(min_time, view_start)
            view_end = min(max_time, view_end)
            
            # Update time scale with new view window
            self.time_scale = TimeScale(view_start, view_end)
            self.time_scrubber.time_scale = self.time_scale
            
            self.logger.info(f"Set initial cursor position to {self.time_scale.time_to_str(latest_time)}")
        
        # Select the first annotation channel by default
        for channel in self.channel_view.channels:
            if channel.annotation_channel:
                channel.selected = True
                self.logger.info(f"Selected channel: {channel.name}")
                # Initialize label editor with the first annotation channel
                self.label_editor.annotation_channel = channel.annotation_channel
                self.label_editor.time_scale = self.time_scale
                break
    
    def _handle_enter(self):
        """Handle enter key to select annotation for editing or finish editing."""
        if self.label_editor.active or self.creating_annotation:
            return
            
        if self.editing_annotation:
            # If already editing, finish editing
            self.editing_annotation = None
            self.editing_edge = None
            self.logger.info("Finished editing annotation")
            return
            
        # Use center time (0.5) instead of current time
        if not self.time_scale:
            return
            
        center_time = self.time_scale.to_scale(0.5)  # Get time at center of display
        selected_channel = None
        
        # Find selected channel
        for channel in self.channel_view.channels:
            if channel.selected and channel.annotation_channel:
                selected_channel = channel
                break
        
        if not selected_channel:
            return
            
        # Find annotation at center time
        annotation = selected_channel.annotation_channel.get_annotation_at_time(center_time)
        if annotation:
            self.editing_annotation = annotation
            self.editing_edge = None  # Reset edge selection
            self.logger.info(f"Selected annotation for editing: {annotation.label}")
            self.logger.info("Press L/R to select left/right edge to adjust")
        else:
            self.editing_annotation = None
            self.logger.warning("No annotation found at center position")
    
    def _handle_l(self):
        """Handle L key to select left edge for editing."""
        if self.editing_annotation and not self.label_editor.active:
            self.editing_edge = 'left'
            self.logger.info("Selected left edge for editing")
    
    def _handle_r(self):
        """Handle R key to select right edge for editing."""
        if self.editing_annotation and not self.label_editor.active:
            self.editing_edge = 'right'
            self.logger.info("Selected right edge for editing")
    
    def _handle_escape(self):
        """Handle escape key press."""
        if self.editing_annotation:
            self.editing_annotation = None
            self.editing_edge = None
            self.logger.info("Cancelled annotation editing")
        elif self.creating_annotation:
            self.creating_annotation = False
            self.annotation_start_time = None
            self.annotation_channel = None
            self.logger.info("Cancelled annotation creation")
        elif self.show_help:
            self.show_help = False
        else:
            self.running = False
    
    def _handle_space(self):
        """Handle space key press."""
        # Get current time (center of display)
        current_time = self.time_scale.to_scale(0.5)  # Always use center of display
        
        # Find selected channel
        selected_channel = None
        for channel in self.channel_view.channels:
            if channel.selected:
                selected_channel = channel
                break
        
        if not selected_channel:
            self.logger.warning("No channel selected.")
            return
            
        if not selected_channel.annotation_channel:
            self.logger.warning("Selected channel does not support annotations.")
            return
        
        # Check if cursor time falls within any existing annotation
        for ann in selected_channel.annotation_channel.annotations:
            if ann.start_time <= current_time <= ann.end_time:
                self.logger.warning(f"Cannot create annotation: cursor overlaps with existing annotation '{ann.label}'")
                return
        
        # Find the annotation that ends just before the cursor
        previous_annotation = None
        for ann in sorted(selected_channel.annotation_channel.annotations, key=lambda a: a.end_time):
            if ann.end_time <= current_time:
                previous_annotation = ann
            else:
                break
        
        # Determine start time
        if previous_annotation:
            start_time = previous_annotation.end_time
        else:
            # No previous annotation, use earliest time from data sources
            earliest_time = None
            for channel in self.channel_view.channels:
                if channel.data_source:
                    source_start, _ = channel.data_source.get_time_range()
                    if earliest_time is None or source_start < earliest_time:
                        earliest_time = source_start
            start_time = earliest_time if earliest_time is not None else self.time_scale.min_time
        
        # Start creating a new annotation
        self.creating_annotation = True
        self.annotation_channel = selected_channel.annotation_channel
        self.annotation_start_time = start_time
        
        # Show label editor for new annotation
        self.label_editor.activate(selected_channel.annotation_channel, self._on_label_submit)
        
        # Log the action
        self.logger.info(f"Creating new annotation from {self.time_scale.time_to_str(start_time)} to {self.time_scale.time_to_str(current_time)}")
    
    def _on_label_submit(self, label_text: str):
        """Handle label submission from the editor."""
        if not self.creating_annotation or not label_text:
            return
            
        # Use the time at the center of the screen as the end time
        current_time = self.time_scale.to_scale(0.5)  # Center of screen
        
        # Create the annotation
        annotation = Annotation(
            label=label_text,
            start_time=self.annotation_start_time,
            end_time=current_time
        )
        
        # Add to channel
        self.annotation_channel.add_annotation(annotation)
        
        # Add the label to possible labels if it's new
        if label_text not in self.annotation_channel.possible_labels:
            self.annotation_channel.add_label(label_text)
        
        # Reset state
        self.creating_annotation = False
        self.annotation_start_time = None
        self.annotation_channel = None
        
        # Log success
        self.logger.success(f"Created annotation: {annotation.label} ({self.time_scale.time_to_str(annotation.start_time)} - {self.time_scale.time_to_str(annotation.end_time)})")
        
        # Force a redraw
        pygame.display.flip()
    
    def _handle_delete(self):
        """Handle delete/backspace key press."""
        if self.label_editor.active:
            return
            
        if not self.editing_annotation:
            self.logger.info("Select an annotation with Enter before deleting")
            return
            
        # Find the channel containing this annotation
        for channel in self.channel_view.channels:
            if (channel.annotation_channel and 
                self.editing_annotation in channel.annotation_channel.annotations):
                # Remove the annotation
                channel.annotation_channel.remove_annotation(self.editing_annotation)
                self.logger.info(f"Deleted annotation: {self.editing_annotation.label}")
                
                # Clear editing state
                self.editing_annotation = None
                self.editing_edge = None
                return
    
    def _handle_zoom_in(self):
        """Handle zoom in."""
        if self.time_scale:
            self.time_scale.zoom(1.5, 0.5)
    
    def _handle_zoom_out(self):
        """Handle zoom out."""
        if self.time_scale:
            self.time_scale.zoom(0.75, 0.5)
    
    def _handle_left(self):
        """Handle left arrow key."""
        if self.label_editor.active:
            return
            
        if self.editing_annotation and self.editing_edge and self.time_scale:
            # Calculate time delta based on current scroll speed
            time_delta = self.time_scale.to_scale(self.current_scroll_speed) - self.time_scale.to_scale(0)
            
            # Move the selected edge left
            if self.editing_edge == 'left':
                new_time = self.editing_annotation.start_time - time_delta
                # Convert new time to screen coordinates
                new_unit = self.time_scale.to_unit(np.array([new_time]))[0]
                
                # Check if new position would be off screen
                if new_unit < 0:
                    # If would go off screen, move timeline instead
                    self.time_scale.pan(self.current_scroll_speed)
                    return
                    
                # Ensure start time doesn't go beyond timeline start or end time
                if new_time >= self.time_scale.min_time and new_time < self.editing_annotation.end_time:
                    self.editing_annotation.start_time = new_time
                    self.logger.info(f"Adjusted start time to {self.time_scale.time_to_str(new_time)}")
            else:  # right edge
                new_time = self.editing_annotation.end_time - time_delta
                # Convert new time to screen coordinates
                new_unit = self.time_scale.to_unit(np.array([new_time]))[0]
                
                # Check if new position would be off screen
                if new_unit < 0:
                    # If would go off screen, move timeline instead
                    self.time_scale.pan(self.current_scroll_speed)
                    return
                    
                # Ensure end time stays after start time
                if new_time > self.editing_annotation.start_time:
                    self.editing_annotation.end_time = new_time
                    self.logger.info(f"Adjusted end time to {self.time_scale.time_to_str(new_time)}")
        else:
            # Regular timeline panning
            if self.time_scale:
                self.time_scale.pan(self.current_scroll_speed)
    
    def _handle_right(self):
        """Handle right arrow key."""
        if self.label_editor.active:
            return
            
        if self.editing_annotation and self.editing_edge and self.time_scale:
            # Calculate time delta based on current scroll speed
            time_delta = self.time_scale.to_scale(self.current_scroll_speed) - self.time_scale.to_scale(0)
            
            # Move the selected edge right
            if self.editing_edge == 'left':
                new_time = self.editing_annotation.start_time + time_delta
                # Convert new time to screen coordinates
                new_unit = self.time_scale.to_unit(np.array([new_time]))[0]
                
                # Check if new position would be off screen
                if new_unit > 1:
                    # If would go off screen, move timeline instead
                    self.time_scale.pan(-self.current_scroll_speed)
                    return
                    
                # Ensure start time stays before end time
                if new_time < self.editing_annotation.end_time:
                    self.editing_annotation.start_time = new_time
                    self.logger.info(f"Adjusted start time to {self.time_scale.time_to_str(new_time)}")
            else:  # right edge
                new_time = self.editing_annotation.end_time + time_delta
                # Convert new time to screen coordinates
                new_unit = self.time_scale.to_unit(np.array([new_time]))[0]
                
                # Check if new position would be off screen
                if new_unit > 1:
                    # If would go off screen, move timeline instead
                    self.time_scale.pan(-self.current_scroll_speed)
                    return
                    
                # Ensure end time doesn't go beyond timeline end
                if new_time <= self.time_scale.max_time:
                    self.editing_annotation.end_time = new_time
                    self.logger.info(f"Adjusted end time to {self.time_scale.time_to_str(new_time)}")
        else:
            # Regular timeline panning
            if self.time_scale:
                self.time_scale.pan(-self.current_scroll_speed)
    
    def _handle_up(self):
        """Handle up arrow key."""
        if self.label_editor.active:
            return
            
        # Find all annotation channels and their original indices
        annotation_channels = [(i, ch) for i, ch in enumerate(self.channel_view.channels) if ch.annotation_channel]
        if not annotation_channels:
            return
            
        num_ann_channels = len(annotation_channels)
        
        # Find the index of the *currently* selected channel within the annotation_channels list
        current_ann_idx = -1
        for i, (idx, _) in enumerate(annotation_channels):
            if idx == self.selected_channel_index:
                current_ann_idx = i
                break
        
        # Calculate the index of the previous annotation channel
        if current_ann_idx == -1:
            # If the current selection isn't an annotation channel, select the last one
            prev_ann_idx = num_ann_channels - 1
        else:
            prev_ann_idx = (current_ann_idx - 1) % num_ann_channels
            
        # Get the original index of the new channel to select
        new_channel_original_idx = annotation_channels[prev_ann_idx][0]
        new_channel = self.channel_view.channels[new_channel_original_idx]
        
        # Update selection state
        if self.selected_channel_index != new_channel_original_idx:
             # Deselect the old one if it exists and is different
            if self.selected_channel_index >= 0 and self.selected_channel_index < len(self.channel_view.channels):
                self.channel_view.channels[self.selected_channel_index].selected = False
            
            self.selected_channel_index = new_channel_original_idx
            new_channel.selected = True
            self.label_editor.annotation_channel = new_channel.annotation_channel
            
            # Update status bar and log
            channel_name = new_channel.name
            self.status_bar.set_channel_info(f"Selected: {channel_name}")
            self.logger.info(f"Selected channel: {channel_name}")
    
    def _handle_down(self):
        """Handle down arrow key."""
        if self.label_editor.active:
            return
            
        # Find all annotation channels and their original indices
        annotation_channels = [(i, ch) for i, ch in enumerate(self.channel_view.channels) if ch.annotation_channel]
        if not annotation_channels:
            return
            
        num_ann_channels = len(annotation_channels)
        
        # Find the index of the *currently* selected channel within the annotation_channels list
        current_ann_idx = -1
        for i in range(num_ann_channels):
            if annotation_channels[i][0] == self.selected_channel_index:
                current_ann_idx = i
                break
        
        # Calculate the index of the next annotation channel
        if current_ann_idx == -1:
            # If the current selection isn't an annotation channel, select the first one
            next_ann_idx = 0
        else:
            next_ann_idx = (current_ann_idx + 1) % num_ann_channels
            
        # Get the original index of the new channel to select
        new_channel_original_idx = annotation_channels[next_ann_idx][0]
        new_channel = self.channel_view.channels[new_channel_original_idx]
        
        # Update selection state
        if self.selected_channel_index != new_channel_original_idx:
            # Deselect the old one if it exists and is different
            if self.selected_channel_index >= 0 and self.selected_channel_index < len(self.channel_view.channels):
                self.channel_view.channels[self.selected_channel_index].selected = False
                
            self.selected_channel_index = new_channel_original_idx
            new_channel.selected = True
            self.label_editor.annotation_channel = new_channel.annotation_channel
            
            # Update status bar and log
            channel_name = new_channel.name
            self.status_bar.set_channel_info(f"Selected: {channel_name}")
            self.logger.info(f"Selected channel: {channel_name}")
    
    def _handle_tab(self):
        """Handle tab key to cycle through annotation channels."""
        # Find the next annotation channel
        start_index = self.selected_channel_index
        next_index = (start_index + 1) % len(self.channel_view.channels)
        
        # Look for the next annotation channel
        while next_index != start_index:
            if self.channel_view.channels[next_index].annotation_channel:
                # Found an annotation channel
                self.channel_view.channels[self.selected_channel_index].selected = False
                self.selected_channel_index = next_index
                self.channel_view.channels[self.selected_channel_index].selected = True
                channel_name = self.channel_view.channels[next_index].name
                self.status_bar.set_channel_info(f"Selected: {channel_name}")
                self.logger.info(f"Selected channel: {channel_name}")
                break
            next_index = (next_index + 1) % len(self.channel_view.channels)
    
    def _handle_help(self):
        """Handle help button click."""
        self.show_help = not self.show_help
        if self.show_help:
            self.logger.info("Showing help (press F1 or Help button to hide)")
        else:
            self.logger.info("Help hidden")
    
    def _handle_c(self):
        """Handle c key to change the label of the selected annotation."""
        if self.label_editor.active or not self.editing_annotation:
            return
            
        # Find the annotation channel this belongs to
        for channel in self.channel_view.channels:
            if (channel.annotation_channel and 
                self.editing_annotation in channel.annotation_channel.annotations):
                # Activate label editor with current label and proper channel
                self.label_editor.text = self.editing_annotation.label
                self.label_editor.cursor_pos = len(self.editing_annotation.label)
                self.label_editor.activate(
                    channel.annotation_channel,  # Pass the channel instead of just the label
                    lambda new_label: self._on_label_change(self.editing_annotation, new_label)
                )
                self.logger.info("Editing label - press Enter to confirm or Escape to cancel")
                break
    
    def _on_label_change(self, annotation: Annotation, new_label: str):
        """Handle label change for an existing annotation."""
        if not new_label:
            self.logger.warning("Label cannot be empty")
            return
            
        old_label = annotation.label
        annotation.label = new_label
        
        # Find the annotation channel this belongs to
        for channel in self.channel_view.channels:
            if (channel.annotation_channel and 
                annotation in channel.annotation_channel.annotations):
                # Add the new label to possible labels if it's new
                channel.annotation_channel.add_label(new_label)
                break
        
        self.logger.success(f"Changed label from '{old_label}' to '{new_label}'")
        
        # Reset editing state but keep annotation selected
        self.editing_edge = None
        self.label_editor.active = False
    
    def _handle_save(self):
        """Handle save button click."""
        try:
            saved_files = []
            for channel in self.channel_view.channels:
                if channel.annotation_channel and channel.annotation_channel.annotations:
                    # Create filename from channel name (replace spaces with underscores)
                    filename = os.path.join(self.annotation_dir, f"{channel.annotation_channel.name.replace(' ', '_')}.csv")
                    
                    # Create DataFrame with annotations
                    annotations_data = {
                        'label': [],
                        'start_time': [],
                        'end_time': []
                    }
                    
                    # Sort annotations by start time
                    sorted_annotations = sorted(channel.annotation_channel.annotations, 
                                             key=lambda a: a.start_time)
                    
                    # Add each annotation to the data
                    for ann in sorted_annotations:
                        annotations_data['label'].append(ann.label)
                        annotations_data['start_time'].append(str(ann.start_time))
                        annotations_data['end_time'].append(str(ann.end_time))
                    
                    # Create and save DataFrame
                    df = pd.DataFrame(annotations_data)
                    df.to_csv(filename, index=False)
                    saved_files.append(filename)
            
            if saved_files:
                self.logger.success(f"Saved annotations to: {', '.join(os.path.basename(f) for f in saved_files)}")
            else:
                self.logger.warning("No annotations to save")
            
        except Exception as e:
            self.logger.error(f"Failed to save annotations: {str(e)}")
    
    def load_annotations(self, channel_name: str = None):
        """Load annotations from CSV file(s).
        
        Args:
            channel_name: Optional channel name. If None, loads all channels.
        """
        try:
            if channel_name:
                # Load specific channel
                filename = os.path.join(self.annotation_dir, f"{channel_name.replace(' ', '_')}.csv")
                if not os.path.exists(filename):
                    self.logger.error(f"Annotation file not found: {filename}")
                    return
                channels_to_load = [(channel_name, filename)]
            else:
                # Load all channels that have corresponding files
                channels_to_load = []
                for channel in self.channel_view.channels:
                    if channel.annotation_channel:
                        filename = os.path.join(self.annotation_dir, 
                                             f"{channel.annotation_channel.name.replace(' ', '_')}.csv")
                        if os.path.exists(filename):
                            channels_to_load.append((channel.annotation_channel.name, filename))
            
            total_loaded = 0
            for channel_name, filename in channels_to_load:
                # Read CSV file
                df = pd.read_csv(filename)
                
                # Find matching channel
                target_channel = None
                for channel in self.channel_view.channels:
                    if (channel.annotation_channel and 
                        channel.annotation_channel.name == channel_name):
                        target_channel = channel
                        break
                
                if not target_channel:
                    self.logger.error(f"Channel not found: {channel_name}")
                    continue
                
                # Clear existing annotations
                target_channel.annotation_channel.annotations = []
                
                # Create new annotations from CSV data
                for _, row in df.iterrows():
                    annotation = Annotation(
                        label=row['label'],
                        start_time=np.datetime64(row['start_time']),
                        end_time=np.datetime64(row['end_time'])
                    )
                    target_channel.annotation_channel.annotations.append(annotation)
                    
                    # Add label to possible labels if it's new
                    if row['label'] not in target_channel.annotation_channel.possible_labels:
                        target_channel.annotation_channel.add_label(row['label'])
                
                # Sort annotations
                target_channel.annotation_channel.sort_annotations()
                total_loaded += len(df)
                self.logger.info(f"Loaded {len(df)} annotations for channel: {channel_name}")
            
            if total_loaded > 0:
                self.logger.success(f"Loaded {total_loaded} total annotations from {len(channels_to_load)} channels")
            else:
                self.logger.warning("No annotations loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load annotations: {str(e)}")
    
    def run(self):
        """Run the annotation tool."""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                
                # Handle window resize
                elif event.type == pygame.VIDEORESIZE:
                    new_width, new_height = event.w, event.h
                    self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                    self._update_layout(new_width, new_height)
                    self.logger.info(f"Window resized to: {new_width}x{new_height}")
                    continue # Skip other event handling for this frame after resize
                
                # Handle Mouse Events
                elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL):
                    # First try channel view
                    if self.channel_view.handle_event(event):
                        continue
                    
                    # Then check for status bar button clicks
                    if event.type == pygame.MOUSEBUTTONDOWN and self.status_bar.rect.collidepoint(event.pos):
                        handled = False
                        # Check left buttons (Save, Help)
                        for button_data in self.status_bar.left_buttons:
                            # Ensure the rect exists (calculated during render)
                            if "rect" in button_data and button_data["rect"].collidepoint(event.pos):
                                button_data["callback"]() # Call the button's function
                                handled = True
                                break # Stop checking buttons once one is clicked
                        if handled:
                            continue # Skip other click handling if a button was pressed
                
                # Handle keyboard events
                elif event.type == pygame.KEYDOWN:
                    # Update key states for continuous scrolling
                    if event.key in self.key_states:
                        self.key_states[event.key] = True
                        self.current_scroll_speed = self.scroll_speed  # Reset to initial slow speed on each press
                    
                    # First try label editor
                    if self.label_editor.active:
                        if self.label_editor.handle_event(event):
                            continue
                    else:
                        # Check for shift+equals (plus) key combination for zoom in
                        if event.key == pygame.K_EQUALS and event.mod & pygame.KMOD_SHIFT:
                            self._handle_zoom_in()
                            continue
                        
                        # Handle other keyboard shortcuts
                        if event.key in self.shortcuts:
                            self.shortcuts[event.key]()
                            continue
                    
                    # Handle other keyboard events
                    if self.time_scrubber.handle_event(event):
                        continue
                    
                    if not self.label_editor.active and self.channel_view.handle_event(event):
                        continue
                
                # Handle key up events for continuous scrolling
                elif event.type == pygame.KEYUP:
                    if event.key in self.key_states:
                        self.key_states[event.key] = False
                        self.current_scroll_speed = self.scroll_speed  # Reset speed when key is released
            
            # Handle continuous scrolling
            current_time = pygame.time.get_ticks()
            dt = current_time - self.last_scroll_time
            self.last_scroll_time = current_time
            
            if not self.label_editor.active and self.time_scale:
                # Update scroll speed based on how long the key has been held
                if self.key_states[pygame.K_LEFT] or self.key_states[pygame.K_RIGHT]:
                    # Gradually increase speed while key is held
                    self.current_scroll_speed = min(
                        self.max_scroll_speed,
                        self.current_scroll_speed + (self.scroll_acceleration * dt)
                    )
                else:
                    self.current_scroll_speed = self.scroll_speed
                
                # Apply scrolling
                if self.key_states[pygame.K_LEFT]:
                    if self.editing_annotation and self.editing_edge:
                        self._handle_left()  # Use existing edge adjustment logic
                    else:
                        self.time_scale.pan(self.current_scroll_speed)
                elif self.key_states[pygame.K_RIGHT]:
                    if self.editing_annotation and self.editing_edge:
                        self._handle_right()  # Use existing edge adjustment logic
                    else:
                        self.time_scale.pan(-self.current_scroll_speed)
            
            # Render
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw grid (adjusted to match data view width)
            grid_width = self.screen_width - 60  # Leave space for axis labels
            grid_height = self.screen_height - self.total_bottom_height - 30  # Leave space for time labels
            for x in range(0, grid_width, 50):
                pygame.draw.line(self.screen, GRID_COLOR, (x + 30, 0), (x + 30, grid_height))
            for y in range(0, grid_height, 50):
                pygame.draw.line(self.screen, GRID_COLOR, (30, y), (grid_width + 30, y))
            
            # Render channel view
            self.channel_view.render(self.screen, self.time_scale)
            
            # Draw editing indicators if an annotation is being edited
            if self.editing_annotation and self.time_scale:
                # Convert times to screen coordinates
                start_unit = self.time_scale.to_unit(np.array([self.editing_annotation.start_time]))[0]
                end_unit = self.time_scale.to_unit(np.array([self.editing_annotation.end_time]))[0]
                
                start_x = int(self.screen.get_width() * start_unit)
                end_x = int(self.screen.get_width() * end_unit)
                
                # Draw edge indicators
                if self.editing_edge == 'left':
                    # Draw left edge indicator (thick)
                    pygame.draw.line(self.screen, HIGHLIGHT_COLOR,
                                   (start_x, 0),
                                   (start_x, self.screen.get_height() - 90),
                                   3)
                    # Draw right edge indicator (thin)
                    pygame.draw.line(self.screen, HIGHLIGHT_COLOR,
                                   (end_x, 0),
                                   (end_x, self.screen.get_height() - 90),
                                   1)
                elif self.editing_edge == 'right':
                    # Draw left edge indicator (thin)
                    pygame.draw.line(self.screen, HIGHLIGHT_COLOR,
                                   (start_x, 0),
                                   (start_x, self.screen.get_height() - 90),
                                   1)
                    # Draw right edge indicator (thick)
                    pygame.draw.line(self.screen, HIGHLIGHT_COLOR,
                                   (end_x, 0),
                                   (end_x, self.screen.get_height() - 90),
                                   3)
                else:
                    # Draw both edges with medium lines when no edge is selected
                    pygame.draw.line(self.screen, HIGHLIGHT_COLOR,
                                   (start_x, 0),
                                   (start_x, self.screen.get_height() - 90),
                                   2)
                    pygame.draw.line(self.screen, HIGHLIGHT_COLOR,
                                   (end_x, 0),
                                   (end_x, self.screen.get_height() - 90),
                                   2)
                    
                # Draw editing help with background
                prompts = [
                    ("Editing", [
                        f"Currently editing: {self.editing_annotation.label}",
                        "L/R: Edit left/right boundary",
                        "C: Change label",
                        "Delete: Remove annotation",
                        "Enter: Finish editing"
                    ])
                ]
                
                font = pygame.font.SysFont("Helvetica Neue", 14)
                line_height = font.get_linesize() + 2
                section_spacing = 15
                padding = 20
                
                # Calculate dimensions for background
                max_width = 0
                total_height = padding * 2
                
                for section_title, lines in prompts:
                    title_surface = font.render(section_title, True, HIGHLIGHT_COLOR)
                    max_width = max(max_width, title_surface.get_width())
                    total_height += title_surface.get_height()
                    
                    for line in lines:
                        text_surface = font.render(line, True, TEXT_COLOR)
                        max_width = max(max_width, text_surface.get_width())
                        total_height += line_height
                
                # Add padding to width
                total_width = max_width + padding * 2
                
                # Create and position background
                bg_rect = pygame.Rect(0, 0, total_width, total_height)
                bg_rect.centerx = self.screen.get_rect().centerx
                bg_rect.top = 40  # Position below the time display
                
                bg_surface = pygame.Surface(bg_rect.size)
                bg_surface.set_alpha(220)
                bg_surface.fill(BACKGROUND_COLOR)
                
                # Draw border
                pygame.draw.rect(bg_surface, GRID_COLOR, bg_surface.get_rect(), 1)
                
                # Draw background
                self.screen.blit(bg_surface, bg_rect.topleft)
                
                # Draw text
                current_y = bg_rect.top + padding
                current_x = bg_rect.left + padding
                
                for section_title, lines in prompts:
                    # Draw section title
                    title_surface = font.render(section_title, True, HIGHLIGHT_COLOR)
                    self.screen.blit(title_surface, (current_x, current_y))
                    current_y += title_surface.get_height()
                    
                    # Draw section lines
                    line_x = current_x + 20
                    for line in lines:
                        text_surface = font.render(line, True, TEXT_COLOR)
                        self.screen.blit(text_surface, (line_x, current_y))
                        current_y += line_height
                
                # Draw the current edge being adjusted and its time
                if self.editing_edge:
                    if self.editing_edge == 'left':
                        time_str = self.time_scale.time_to_str(self.editing_annotation.start_time)
                        x_pos = start_x
                    else:
                        time_str = self.time_scale.time_to_str(self.editing_annotation.end_time)
                        x_pos = end_x
                    
                    text_surface = font.render(time_str, True, HIGHLIGHT_COLOR)
                    text_x = max(10, min(x_pos - text_surface.get_width() // 2,
                                       self.screen.get_width() - text_surface.get_width() - 10))
                    self.screen.blit(text_surface, (text_x, bg_rect.bottom + 10))
            
            # Draw center time indicator (main cursor line)
            # Adjust grid_width calculation to use the actual channel view width
            channel_view_width = self.channel_view.rect.width
            center_x = self.channel_view.rect.left + (channel_view_width // 2)
            
            # Draw the line from top to bottom of screen
            pygame.draw.line(self.screen, CURSOR_COLOR, 
                           (center_x, 0),  # Start from top
                           (center_x, self.screen_height),  # End at bottom of screen
                           1)  # Thinner line (was 2)
            
            # Calculate and draw the timestamp for the center cursor position
            if self.time_scale:
                # Calculate the time at the center of the channel view
                center_unit = (center_x - self.channel_view.rect.left) / channel_view_width
                center_time = self.time_scale.to_scale(center_unit)
                
                if center_time:
                    cursor_font = pygame.font.SysFont("Helvetica Neue", 14)
                    time_text = self.time_scale.time_to_str(center_time)
                    cursor_label = cursor_font.render(time_text, True, HIGHLIGHT_COLOR)
                    
                    # Center label horizontally on cursor line, but keep within screen bounds
                    label_x = max(5, 
                                min(center_x - cursor_label.get_width() // 2,
                                    self.screen_width - cursor_label.get_width() - 5))
                    
                    # Position cursor time at the top of the screen
                    cursor_label_y = 5
                    
                    # Draw semi-transparent background for better readability
                    label_bg_rect = pygame.Rect(
                        label_x - 2, cursor_label_y - 2, 
                        cursor_label.get_width() + 4, cursor_label.get_height() + 4
                    )
                    bg_surface = pygame.Surface((label_bg_rect.width, label_bg_rect.height))
                    bg_surface.set_alpha(180)
                    bg_surface.fill(BACKGROUND_COLOR)
                    self.screen.blit(bg_surface, label_bg_rect)
                    
                    # Draw the current time label
                    self.screen.blit(cursor_label, (label_x, cursor_label_y))
            
            # Render time scrubber (now only shows window start/end times)
            if self.time_scrubber:
                self.time_scrubber.render(self.screen)
            
            # Render help screen
            if self.show_help:
                self._render_help()
            
            # Render label editor
            self.label_editor.render(self.screen)
            
            # Render status bar
            if self.status_bar:
                self.status_bar.render(self.screen)
            
            pygame.display.flip()
            self.clock.tick(FPS)
    
    def _render_help(self):
        """Render help text with a background."""
        help_sections = [
            ("Navigation", [
                "Left/Right: Pan timeline (when not editing)",
                "Up/Down: Select channel (when not editing)",
                "Tab: Cycle through annotation channels (when not editing)",
                "=/- or +/-: Zoom in/out",
                "Escape: Cancel current action or quit",
                "F1: Toggle help"
            ]),
            ("Annotation", [
                "Space: Create new annotation",
                "Enter: Select annotation at cursor / Finish editing",
                "L/R: Select left/right edge of selected annotation",
                "C: Change label",
                "Left/Right: Move selected edge",
                "Delete/Backspace: Remove annotation",
                "When editing labels:",
                "  Up/Down: Navigate suggestions",
                "  Tab: Complete with selected suggestion",
                "  Enter: Accept and create annotation",
                "  Escape: Cancel editing"
            ])
        ]
        
        font = pygame.font.SysFont("Helvetica Neue", 14) # Reduced from 20 to 14
        line_height = font.get_linesize() + 2 # Add small spacing between lines
        section_spacing = 15 # Spacing between sections
        padding = 20 # Padding around the text block
        
        # --- Calculate dimensions needed for the background --- 
        max_width = 0
        total_height = padding * 2 # Top and bottom padding
        
        for i, (section_title, lines) in enumerate(help_sections):
            title_surface = font.render(section_title, True, HIGHLIGHT_COLOR)
            max_width = max(max_width, title_surface.get_width())
            total_height += title_surface.get_height()
            
            for line in lines:
                text_surface = font.render(line, True, TEXT_COLOR)
                max_width = max(max_width, text_surface.get_width())
                total_height += line_height
            
            if i < len(help_sections) - 1:
                total_height += section_spacing
                
        # Add side padding to max_width
        total_width = max_width + padding * 2

        # --- Create and position the background surface --- 
        bg_rect = pygame.Rect(0, 0, total_width, total_height)
        bg_rect.center = self.screen.get_rect().center # Center it on the screen
        
        bg_surface = pygame.Surface(bg_rect.size)
        bg_surface.set_alpha(220) # Semi-transparent
        bg_surface.fill(BACKGROUND_COLOR) 
        
        # Draw border for the background
        pygame.draw.rect(bg_surface, GRID_COLOR, bg_surface.get_rect(), 1)
        
        # Blit the background onto the main screen
        self.screen.blit(bg_surface, bg_rect.topleft)
        
        # --- Draw the help text on top of the background --- 
        current_y = bg_rect.top + padding # Start drawing text inside the background
        current_x = bg_rect.left + padding
        
        for section_title, lines in help_sections:
            # Draw section title
            title_surface = font.render(section_title, True, HIGHLIGHT_COLOR)
            self.screen.blit(title_surface, (current_x, current_y))
            current_y += title_surface.get_height()
            
            # Draw section lines
            line_x = current_x + 20 # Indent lines slightly
            for line in lines:
                text_surface = font.render(line, True, TEXT_COLOR)
                self.screen.blit(text_surface, (line_x, current_y))
                current_y += line_height
            
            current_y += section_spacing


def main():
    """Main entry point."""
    # Create and run the tool with default settings
    tool = AnnotationTool()
    tool.run()

if __name__ == "__main__":
    main()