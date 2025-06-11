import base64
import hashlib
import io
import json
import os
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
from PIL import Image


def get_concentration_color(concentration):
    """Get color for concentration level using ordinal scale"""
    # Ordinal color scale from light blue (0) to dark red (80)
    color_map = {
        0: "#e6f3ff",  # Very light blue (control)
        5: "#b3d9ff",  # Light blue
        10: "#80bfff",  # Medium blue
        20: "#4da6ff",  # Blue
        40: "#ff8000",  # Orange
        80: "#ff0000",  # Red
    }
    return color_map.get(concentration, "#808080")  # Gray for unknown


def get_concentration_color_plotly(concentration):
    """Get plotly-compatible color for concentration level"""
    color_map = {
        0: "#e6f3ff",  # Very light blue (control)
        5: "#b3d9ff",  # Light blue
        10: "#80bfff",  # Medium blue
        20: "#4da6ff",  # Blue
        40: "#ff8000",  # Orange
        80: "#ff0000",  # Red
    }
    return color_map.get(concentration, "#808080")  # Gray for unknown


# Parse uploaded image
def parse_image(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return np.array(Image.open(io.BytesIO(decoded)))


# Convert numpy array to base64 image
def numpy_to_b64(img_array):
    if len(img_array.shape) == 2:
        # Convert grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)

    # Normalize if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    # Convert to PIL Image
    img = Image.fromarray(img_array)

    # Save to bytes buffer
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Convert to base64
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"


# Calculate image hash for caching
def calculate_image_hash(img_array):
    return hashlib.md5(img_array.tobytes()).hexdigest()


# Simple file-based cache system
CACHE_DIR = "cache"
CROPPED_CACHE_DIR = os.path.join(CACHE_DIR, "cropped")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CROPPED_CACHE_DIR, exist_ok=True)


# Check if result is in cache
def get_from_cache(img_hash):
    cache_file = os.path.join(CACHE_DIR, f"{img_hash}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading from cache: {e!s}")
    return None


# Save results to cache
def save_to_cache(img_hash, cell_data, encoded_image, mask_data, predicted_properties, cropped_images=None):
    cache_file = os.path.join(CACHE_DIR, f"{img_hash}.json")
    try:
        cache_data = {
            "cell_data": cell_data,
            "encoded_image": encoded_image,
            "mask_data": mask_data,
            "predicted_properties": predicted_properties,
        }

        # Don't include cropped images in the main cache file as they can be large
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Save cropped images separately - Fixed loop variable overwriting (PLW2901)
        if cropped_images:
            for cell_id, img_data in cropped_images.items():
                crop_file = os.path.join(CROPPED_CACHE_DIR, f"{img_hash}_{cell_id}.png")
                # Save as PNG file
                with open(crop_file, "wb") as f:
                    # Remove data URL prefix if present - Fixed variable overwriting
                    processed_data = img_data.split(",", 1)[1] if "," in img_data else img_data
                    f.write(base64.b64decode(processed_data))

        return True
    except Exception as e:
        print(f"Error saving to cache: {e!s}")
        return False


# Get cropped image from cache
def get_cropped_image(img_hash, cell_id):
    crop_file = os.path.join(CROPPED_CACHE_DIR, f"{img_hash}_{cell_id}.png")
    if os.path.exists(crop_file):
        try:
            with open(crop_file, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
                return f"data:image/png;base64,{img_data}"
        except Exception as e:
            print(f"Error reading cropped image from cache: {e!s}")
    return None


def update_full_figure_layout(fig: go.Figure, width, height, has_cell_data=False, show_annotations=True):
    # Update layout - we need fixed pixel coordinates, not aspect ratio preservation
    fig.update_layout(
        autosize=True,
        height=600,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        clickmode="event",
        # Add helpful annotation
        annotations=[
            {
                "text": "Click on any colored circle to view cell details",
                "x": 0.5,
                "y": 0.01,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
                "bgcolor": "rgba(255,255,255,0.7)",
                "bordercolor": "gray",
                "borderwidth": 1,
                "borderpad": 4,
                "visible": bool(has_cell_data and show_annotations),
            }
        ]
        if has_cell_data
        else [],
    )

    # Update axes
    fig.update_xaxes(
        range=[0, width],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        constrain="domain",  # This helps maintain image dimensions
    )
    fig.update_yaxes(
        range=[height, 0],  # Reverse y-axis to match image coordinates
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="x",  # Preserve aspect ratio
        scaleratio=1.0,  # 1:1 aspect ratio
    )

    return fig


# Create visualization with all elements
def create_complete_figure(encoded_image, cell_data=None, mask_data=None, show_annotations=True):
    """Create a complete figure with all elements, with annotations visible based on show_annotations"""

    try:
        # Extract image dimensions from encoded image
        content_type, content_string = encoded_image.split(",")
        decoded = base64.b64decode(content_string)
        img = np.array(Image.open(io.BytesIO(decoded)))

        # Get image dimensions
        height, width = img.shape[:2]

        # Create the base figure with the original image
        fig = go.Figure()

        # Add the original image as the base layer
        fig.add_layout_image(
            {
                "source": encoded_image,
                "xref": "x",
                "yref": "y",
                "x": 0,
                "y": 0,
                "sizex": width,
                "sizey": height,
                "sizing": "stretch",  # Use stretch for pixel-perfect mapping
                "opacity": 1,
                "layer": "below",
            }
        )

        # Add cell overlays if data exists
        if cell_data and mask_data:
            # Add boundary points
            boundary_points = np.array(np.where(np.array(mask_data["boundary"]))).T
            if len(boundary_points) > 0:
                boundary_y, boundary_x = boundary_points[:, 0], boundary_points[:, 1]
                fig.add_trace(
                    go.Scatter(
                        x=boundary_x,
                        y=boundary_y,
                        mode="markers",
                        marker={"color": "white", "size": 1},
                        hoverinfo="none",
                        showlegend=False,
                        visible=show_annotations,
                    )
                )

            # Add transparent cell centroids for clicking
            for cell in cell_data:
                predicted_props = cell.get("predicted_properties", {})

                # Color cells based on concentration level (ordinal)
                if "concentration" in predicted_props:
                    concentration = predicted_props.get("concentration", 0)
                    color = get_concentration_color(concentration)
                else:
                    # Fallback to size-based coloring
                    is_large = cell["is_large"]
                    color = "green" if is_large else "red"

                # Create hover text with predicted properties
                hover_lines = [f"<b>Cell {cell['id']}</b>"]

                if "predicted_class" in predicted_props:
                    hover_lines.append(f"Class: {predicted_props['predicted_class']}")
                    hover_lines.append(f"Confidence: {predicted_props.get('confidence', 0):.2f}")
                    concentration = predicted_props.get("concentration", 0)
                    hover_lines.append(f"Concentration: {concentration}")
                else:
                    hover_lines.append(f"Size: {int(cell['area'])} pixels")

                hover_lines.append("<b>Click for details</b>")
                hover_text = "<br>".join(hover_lines)

                # Add centroid marker with good click/hover area
                fig.add_trace(
                    go.Scatter(
                        x=[cell["centroid_x"]],
                        y=[cell["centroid_y"]],
                        mode="markers",
                        marker={
                            "size": max(20, np.sqrt(cell["area"]) / 3),  # Larger markers for better visibility
                            "color": color,
                            "opacity": 0.7,  # More visible
                            "line": {"width": 2, "color": "white"},  # Add white border
                            "symbol": "circle",  # Clear circle shape
                        },
                        name=f"Cell {cell['id']}",
                        hoverinfo="text",
                        hovertext=hover_text,
                        customdata=[cell["id"]],  # Store cell ID for click events
                        visible=show_annotations,
                    )
                )

                # Add cell ID and prediction as text annotation
                if "predicted_class" in predicted_props:
                    # Show cell ID and concentration
                    concentration = predicted_props.get("concentration", 0)
                    annotation_text = f"{cell['id']}\n{concentration}"
                else:
                    # Fallback to cell ID and size classification
                    size_class = "L" if cell["is_large"] else "S"
                    annotation_text = f"{cell['id']}\n{size_class}"

                fig.add_trace(
                    go.Scatter(
                        x=[cell["centroid_x"]],
                        y=[cell["centroid_y"]],
                        mode="text",
                        text=annotation_text,
                        textfont={"color": "white", "size": 9, "family": "Arial Black"},
                        hoverinfo="none",
                        showlegend=False,
                        visible=show_annotations,
                    )
                )

        update_full_figure_layout(fig, width, height, cell_data is not None, show_annotations)

        return fig

    except Exception as e:
        print(f"Error creating complete figure: {e!s}")
        return None


# Create cropped image
def create_cell_crop(encoded_image, cell, mask_data, padding=10):
    # Get bounding box with padding
    y0, x0, y1, x1 = cell["bbox"]
    padding = 10
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(len(mask_data["mask"]), y1 + padding)
    x1 = min(len(mask_data["mask"][0]), x1 + padding)

    # Get the full image
    content_type, content_string = encoded_image.split(",")
    decoded = base64.b64decode(content_string)
    img = np.array(Image.open(io.BytesIO(decoded)))

    # Create a cropped version
    img_cropped = img[y0:y1, x0:x1]

    # Convert to base64
    return numpy_to_b64(img_cropped)
