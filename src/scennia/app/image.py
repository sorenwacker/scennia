import base64
import hashlib
import io
from timeit import default_timer as timer

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from PIL.ImageFile import ImageFile

from scennia.app.data import (
    EncodedImage,
    ImageData,
    ProcessedData,
    confidence_into_english,
    relative_lactate_concentration_into_resistance,
)

# Simple color scale from blue - white - red.
BLUE_WHITE_RED_COLOR_SCALE = [
    "rgb(64,64,255)",
    "rgb(245,245,245)",
    "rgb(255,64,64)",
]


# Get the color of the cell based on its lactate concentration
def lactate_concentration_color(concentration: int) -> tuple[int, int, int]:
    color_map = {
        0: "#c0c6ca",
        5: "#9bb9d7",
        10: "#80c0ff",
        20: "#4da6ff",
        40: "#ff8000",
        80: "#ff0000",
    }
    color_hex = color_map.get(concentration, "#808080")
    (r, g, b) = px.colors.hex_to_rgb(color_hex)
    return (r, g, b)


# Get the color of the cell based on its relative lactate concentration
def relative_lactate_concentration_color(r_concentration: int) -> tuple[int, int, int]:
    # Plotly color scales need the value to be in range 0-1, so we need to scale the relative concentration first.
    # Scale from -50-50 to 0-1. Note that the actual range is from -80 to 80, but scaling to -50-50 exagerates
    # the colors a bit.
    scaled = (50.0 + float(r_concentration)) / 100.0
    # Clamp the scaled value to 0-1
    scaled = max(0.0, scaled)  # Clamp negative range
    scaled = min(1.0, scaled)  # Clamp positive range
    # Interpolate a color from the color scale based on the scaled value
    color_label = px.colors.sample_colorscale(BLUE_WHITE_RED_COLOR_SCALE, scaled)[0]
    (r, g, b) = px.colors.unlabel_rgb(color_label)  # For some reason this turns the ints into floats...
    return (int(r), int(g), int(b))  # So turn them back into ints here


# Get the color of the cell based on its lactate resistance.
def lactate_resistance_color(lactate_resistance: str) -> tuple[int, int, int]:
    if lactate_resistance == "very likely lactate resistant":
        return (64, 64, 240)
    if lactate_resistance == "likely lactate resistant":
        return (170, 170, 240)
    return (240, 64, 64)


# Try to color based on relative lactate concentration
def concentration_color(concentration: int | None, r_concentration: int | None) -> tuple[int, int, int]:
    if r_concentration is not None:
        return relative_lactate_concentration_color(r_concentration)
    if concentration is not None:
        return lactate_concentration_color(concentration)
    return (127, 127, 127)  # Fallback: grey


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


# Update image analysis figure layout
def update_image_analysis_figure_layout(
    fig: go.Figure, width: int, height: int, has_cell_data=False, show_segmentation=True, show_classification=True
):
    # Update layout - we need fixed pixel coordinates, not aspect ratio preservation
    fig.update_layout(
        autosize=True,
        height=650,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        clickmode="event",
        # Add helpful annotation
        annotations=[
            {
                "text": "Click on a cell to view its details",
                "x": 0.5,
                "y": 0.01,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
                "bgcolor": "rgba(255,255,255,0.7)",
                "bordercolor": "gray",
                "borderwidth": 1,
                "borderpad": 2,
                "visible": bool(has_cell_data and (show_segmentation or show_classification)),
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


# Create image analysis figure
def create_image_analysis_figure(encoded_image: EncodedImage) -> tuple[EncodedImage, go.Figure]:
    # Create a figure with just the original image
    figure_start = timer()
    figure = go.Figure()
    figure.add_layout_image({
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
    update_image_analysis_figure_layout(figure, encoded_image.width, encoded_image.height, False, False)
    dash.callback_context.record_timing("figure", timer() - figure_start, "Create figure")

    return encoded_image, figure


# Create processed image analysis figure
def create_processed_image_analysis_figure(
    image_data: ImageData,
    processed_data: ProcessedData,
    show_segmentation=True,
    show_classification=True,
):
    """Create a complete figure with all elements, with annotations visible based on show_segmentation"""

    # Create the base figure with the original image
    figure_start = timer()
    figure = go.Figure()

    # Add the original image as the base layer
    encoded_image = image_data.encoded_image
    figure.add_layout_image({
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
    cells = processed_data.cells
    for id, cell in cells.items():
        # Get predicted and relative lactate concentration, along with the confidence of the prediction.
        (concentration, r_concentration, confidence) = cell.lactate_concentration(
            image_data.actual_lactate_concentration()
        )

        # Create hover text with predicted properties
        hover_lines = [f"<b>Cell {id}</b>"]
        if cell.predicted_properties:
            if concentration is not None:
                hover_lines.append(f"Lactate concentration: {concentration}mM")
            if r_concentration is not None:
                hover_lines.append(f"Relative lactate concentration: {r_concentration}mM")
            if confidence is not None:
                hover_lines.append(f"Confidence: {confidence_into_english(confidence)}")
            if r_concentration is not None:
                conclusion = relative_lactate_concentration_into_resistance(r_concentration)
                hover_lines.append(f"<b>Conclusion:<br>  {conclusion}</b>")
        hover_lines.append("<br>Click for more info")
        hover_text = "<br>".join(hover_lines)

        # Try to color based on relative lactate concentration if classification is enabled.
        color = concentration_color(concentration, r_concentration) if show_classification else (255, 255, 255)

        # Draw contours around cells with hover text and click events.
        if len(cell.contour) > 0:
            (r, g, b) = color
            y, x = cell.contour
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.1)",
                    line={"color": px.colors.label_rgb(color)},
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

    update_image_analysis_figure_layout(
        figure, encoded_image.width, encoded_image.height, cells is not None, show_segmentation, show_classification
    )
    dash.callback_context.record_timing("figure", timer() - figure_start, "Create figure")

    return figure
