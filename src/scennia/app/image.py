import base64
import hashlib
import io
import json
import os
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from PIL.ImageFile import ImageFile
from plotly.colors import hex_to_rgb


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


# Decode image from base64
def decode_image(contents: str) -> ImageFile:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))


@dataclass
class EncodedImage:
    contents: str
    width: int
    height: int


# Encode image to base64
def encode_image(image: Image.Image) -> EncodedImage:
    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="WebP")
    buffer.seek(0)
    # Convert to base64
    contents = f"data:image/webp;base64,{base64.b64encode(buffer.read()).decode()}"
    return EncodedImage(contents, image.width, image.height)


# Calculate image hash for caching
def calculate_image_hash(image: ImageFile) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


# Simple file-based cache system
CACHE_DIR = "cache"
CROPPED_CACHE_DIR = os.path.join(CACHE_DIR, "cropped")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CROPPED_CACHE_DIR, exist_ok=True)


# Check if result is in cache
def get_processed_from_cache(image_hash: str):
    cache_file = os.path.join(CACHE_DIR, f"{image_hash}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading from cache: {e!s}")
    return None


# Save results to cache
def save_processed_to_cache(
    img_hash: str, cell_data, mask_data, predicted_properties, cropped_uncompressed_images: dict[str, Image.Image]
):
    cache_file = os.path.join(CACHE_DIR, f"{img_hash}.json")
    try:
        cache_data = {
            "cell_data": cell_data,
            "mask_data": mask_data,
            "predicted_properties": predicted_properties,
            # Don't include images in the main cache file as they can be large
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Compress and save cropped images separately - Fixed loop variable overwriting (PLW2901)
        for cell_id, image in cropped_uncompressed_images.items():
            path = os.path.join(CROPPED_CACHE_DIR, f"{img_hash}_{cell_id}.webp")
            image.save(path)

        return True
    except Exception as e:
        print(f"Error saving to cache: {e!s}")
        return False


# Get compressed cropped image from cache
def get_compressed_cropped_image(image_hash: str, cell_id: str) -> ImageFile | None:
    crop_file = os.path.join(CROPPED_CACHE_DIR, f"{image_hash}_{cell_id}.webp")
    if os.path.exists(crop_file):
        try:
            return Image.open(crop_file)
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
def create_complete_figure(encoded_image: EncodedImage, cell_data=None, show_annotations=True):
    """Create a complete figure with all elements, with annotations visible based on show_annotations"""

    try:
        # Create the base figure with the original image
        fig = go.Figure()

        # Add the original image as the base layer
        fig.add_layout_image({
            "source": encoded_image.contents,
            "xref": "x",
            "yref": "y",
            "x": 0,
            "y": 0,
            "sizex": encoded_image.width,
            "sizey": encoded_image.height,
            "sizing": "stretch",  # Use stretch for pixel-perfect mapping
            "opacity": 1,
            "layer": "below",
        })

        # Add cell overlays if data exists
        if cell_data:
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

                # Draw contours around cells with hover text and click events.
                if len(cell["contour"]) > 0:
                    (r, g, b) = hex_to_rgb(color)
                    y, x = cell["contour"]
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            fill="toself",
                            fillcolor=f"rgba({r},{g},{b},0.1)",
                            line={"color": color},
                            hoveron="fills",
                            hoverinfo="text",
                            hoverlabel={"bgcolor": f"rgba({r},{g},{b},0.5)"},
                            hovertext=hover_text,
                            # We need to use name instead of hovertext when using hoveron="fills".
                            # See: https://stackoverflow.com/a/57937013
                            name=hover_text,
                            customdata=[cell["id"]],  # Store cell ID for click events
                            showlegend=False,
                            visible=show_annotations,
                        )
                    )

        update_full_figure_layout(
            fig, encoded_image.width, encoded_image.height, cell_data is not None, show_annotations
        )

        return fig

    except Exception as e:
        print(f"Error creating complete figure: {e!s}")
        return None


# Create a cell crop, returning an uncompressed cropped image of the cell.
def crop_cell(image: ImageFile, bbox: tuple[float, float, float, float], padding=10) -> Image.Image:
    # Get bounding box with padding
    y0, x0, y1, x1 = bbox
    padding = 10
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(image.height, y1 + padding)
    x1 = min(image.width, x1 + padding)

    # Create a cropped version
    image_array = np.asarray(image)
    img_cropped = image_array[y0:y1, x0:x1]
    return Image.fromarray(img_cropped)
