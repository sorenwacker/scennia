import base64
import hashlib
import io
import json
import os

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from PIL.ImageFile import ImageFile


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


# Decode image from base64
def decode_image(contents: str) -> ImageFile:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))


# Calculate image hash for caching
def calculate_image_hash(image: ImageFile) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


# Simple file-based cache system
CACHE_DIR = "cache"
CROPPED_CACHE_DIR = os.path.join(CACHE_DIR, "cropped")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CROPPED_CACHE_DIR, exist_ok=True)


# Get compressed image from cache
def get_compressed_image_from_cache(image_hash: str) -> ImageFile | None:
    path = os.path.join(CACHE_DIR, f"{image_hash}.webp")
    if os.path.exists(path):
        try:
            return Image.open(path)
        except Exception as e:
            print(f"Error reading compressed image from cache: {e!s}")
    return None


# Save image to cache as original format and compressed format
def save_image_to_cache(image_hash: str, image: ImageFile):
    try:
        image.save(os.path.join(CACHE_DIR, f"{image_hash}.webp"))
    except Exception as e:
        print(f"Error saving image to cache: {e!s}")
        return False


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
def create_complete_figure(image: ImageFile, cell_data=None, mask_data=None, show_annotations=True):
    """Create a complete figure with all elements, with annotations visible based on show_annotations"""

    try:
        # Create the base figure with the original image
        fig = go.Figure()

        # Add the original image as the base layer
        fig.add_layout_image(
            {
                "source": image,
                "xref": "x",
                "yref": "y",
                "x": 0,
                "y": 0,
                "sizex": image.width,
                "sizey": image.height,
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

        update_full_figure_layout(fig, image.width, image.height, cell_data is not None, show_annotations)

        return fig

    except Exception as e:
        print(f"Error creating complete figure: {e!s}")
        return None


# Create a cell crop, returning an uncompressed cropped image of the cell.
def crop_cell(image: ImageFile, cell, mask_data, padding=10) -> Image.Image:
    # Get bounding box with padding
    y0, x0, y1, x1 = cell["bbox"]
    padding = 10
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(len(mask_data["mask"]), y1 + padding)
    x1 = min(len(mask_data["mask"][0]), x1 + padding)

    # Create a cropped version
    image_array = np.asarray(image)
    img_cropped = image_array[y0:y1, x0:x1]
    return Image.fromarray(img_cropped)
