import base64
import hashlib
import io

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from PIL.ImageFile import ImageFile
from plotly.colors import hex_to_rgb

from scennia.app.data import Cell, EncodedImage


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


# Encode image to base64
def encode_image(image: Image.Image) -> EncodedImage:
    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="WebP")
    buffer.seek(0)
    # Convert to base64
    contents = f"data:image/webp;base64,{base64.b64encode(buffer.read()).decode()}"
    return EncodedImage(contents=contents, width=image.width, height=image.height)


# Calculate image hash for caching
def calculate_image_hash(image: ImageFile) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def update_full_figure_layout(fig: go.Figure, width, height, has_cell_data=False, show_segmentation=True):
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
                "text": "Click on a colored cell to view its details",
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
                "visible": bool(has_cell_data and show_segmentation),
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
        fixedrange=True,  # Disabled: zoom, doesn't seem to work well
        constrain="domain",  # This helps maintain image dimensions
    )
    fig.update_yaxes(
        range=[height, 0],  # Reverse y-axis to match image coordinates
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        fixedrange=True,  # Disabled: zoom, doesn't seem to work well
        scaleanchor="x",  # Preserve aspect ratio
        scaleratio=1.0,  # 1:1 aspect ratio
    )

    return fig


# Create visualization with all elements
def create_complete_figure(encoded_image: EncodedImage, cells: list[Cell] | None, show_segmentation=True):
    """Create a complete figure with all elements, with annotations visible based on show_segmentation"""

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
        if cells:
            for cell in cells:
                predicted_props = cell.predicted_properties

                # Color cells based on concentration level (ordinal)
                if "concentration" in predicted_props:
                    concentration = predicted_props.get("concentration", 0)
                    color = get_concentration_color(concentration)
                else:
                    # Fallback to size-based coloring
                    is_large = cell.is_large
                    color = "green" if is_large else "red"

                # Create hover text with predicted properties
                hover_lines = [f"<b>Cell {cell.id}</b>"]

                if "predicted_class" in predicted_props:
                    hover_lines.append(f"Class: {predicted_props['predicted_class']}")
                    hover_lines.append(f"Confidence: {predicted_props.get('confidence', 0):.2f}")
                    concentration = predicted_props.get("concentration", 0)
                    hover_lines.append(f"Concentration: {concentration}")
                else:
                    hover_lines.append(f"Size: {int(cell.area)} pixels")

                hover_lines.append("<b>Click for details</b>")
                hover_text = "<br>".join(hover_lines)

                # Draw contours around cells with hover text and click events.
                if len(cell.contour) > 0:
                    (r, g, b) = hex_to_rgb(color)
                    y, x = cell.contour
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
                            customdata=[cell.id],  # Store cell ID for click events
                            showlegend=False,
                            visible=show_segmentation,
                        )
                    )

        update_full_figure_layout(fig, encoded_image.width, encoded_image.height, cells is not None, show_segmentation)

        return fig

    except Exception as e:
        print(f"Error creating complete figure: {e!s}")
        return None


# Create a cell crop, returning an uncompressed cropped image of the cell.
def crop_cell(image: ImageFile, bbox: list[int], padding=10) -> Image.Image:
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
