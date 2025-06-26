import traceback

import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, ValidationError

from scennia.app.data import (
    Cell,
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


# Image analysis figure


class ImageAnalysisFilter(BaseModel):
    # Only show cells with an area between this area. Defaults to `None``.
    area: tuple[float, float] | None = None
    # Only show cells with this lactate concentration. Defaults to `None``.
    concentration: int | None = None
    # Only show cells with this lactate resistance. Defaults to `None`.`
    resistance: str | None = None

    @classmethod
    def from_json(cls, json_data: str | bytes | bytearray) -> "ImageAnalysisFilter":
        try:
            return ImageAnalysisFilter.model_validate_json(json_data)
        except ValidationError:
            print(f"Failed to parse `ImageAnalysisFilter` from JSON data: '{json_data}'; returning empty filter")
            traceback.print_exc()
        return ImageAnalysisFilter()

    def to_json(self) -> str:
        return ImageAnalysisFilter.model_dump_json(self, indent=2)

    def should_filter_by_area(self, cell: Cell) -> bool:
        if self.area is not None:
            min_area, max_area = self.area
            if cell.area_um < min_area or cell.area_um > max_area:
                return True
        return False

    def should_filter_by_concentration(self, concentration: int) -> bool:
        return bool(self.concentration is not None and concentration != self.concentration)

    def should_filter_by_resistance(self, resistance: str) -> bool:
        return bool(self.resistance is not None and resistance != self.resistance)

    def is_filtering(self) -> bool:
        return self.area is not None or self.concentration is not None or self.resistance is not None

    def to_facet_strs(self) -> list[str]:
        filters = []
        if self.area is not None:
            filters.append(f"area {self.area[0]:.1f} - {self.area[1]:.1f}")
        if self.concentration is not None:
            filters.append(f"concentration {self.concentration}")
        if self.resistance is not None:
            filters.append(f"resistance {self.resistance}")
        return filters


# Create image analysis figure
def create_image_analysis_figure(
    image_data: ImageData,
    processed_data: ProcessedData,
    filter: ImageAnalysisFilter,
    show_segmentation: bool = True,
    show_classification: bool = True,
) -> go.Figure:
    """Create an image analysis figure,
    Args:
        image_data (ImageData): Image data used to create the figure.
        processed_data (ProcessedData): Processed data used to create the figure.
        filter (ImageAnalysisFilter): Filter used to filter cells in the figure.
        show_segmentation (bool): Whether segmentation contours around cells are shown.
        show_classification (bool): Whether cell contours are colored by their predicted classification.
    Returns:
        go.Figure: Image analysis figure
    """

    # Create the base figure with the original image
    fig = go.Figure()
    encoded_image = image_data.encoded_image
    add_layout_image(fig, encoded_image)

    # Add cell overlays
    for id, cell in processed_data.cells.items():
        if filter.should_filter_by_area(cell):
            continue  # Skip if filtered by area

        # Get predicted and relative lactate concentration, along with the confidence of the prediction.
        (concentration, r_concentration, confidence) = cell.lactate_concentration(
            image_data.actual_lactate_concentration()
        )

        if concentration is not None and filter.should_filter_by_concentration(concentration):
            continue  # Skip if filtered by concentration

        # Create hover text with predicted properties
        hover_lines = [f"<b>Cell {id}</b>"]
        if concentration is not None:
            hover_lines.append(f"Predicted Lactate Concentration: {concentration} mM")
        if r_concentration is not None:
            hover_lines.append(f"Predicted Relative to Actual Concentration: {r_concentration} mM")
        if confidence is not None:
            hover_lines.append(f"Confidence: {confidence_into_english(confidence)}")
        if r_concentration is not None:
            resistance = relative_lactate_concentration_into_resistance(r_concentration)
            if filter.should_filter_by_resistance(resistance):
                continue  # Skip if filtered by resistance
            hover_lines.append(f"<b>Conclusion: {resistance}</b>")
        hover_lines.append("<br>Click for more info")
        hover_text = "<br>".join(hover_lines)

        # Try to color based on relative lactate concentration if classification is enabled.
        color = concentration_color(concentration, r_concentration) if show_classification else (255, 255, 255)

        # Draw contours around cells with hover text and click events.
        if len(cell.contour) > 0:
            (r, g, b) = color
            y, x = cell.contour
            fig.add_trace(
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

    # Update layout
    show_annotation = len(processed_data.cells) > 0 and (show_segmentation or show_classification)
    update_image_analysis_figure_layout(fig, encoded_image.width, encoded_image.height, show_annotation)

    return fig


def create_placeholder_image_analysis_figure(encoded_image: EncodedImage) -> go.Figure:
    """Create a placeholder image analysis figure.
    Args:
        encoded_image (EncodedImage): Encoded image to add as layout image.
    Returns:
        go.Figure: Placeholder image analysis figure.
    """
    fig = go.Figure()
    add_layout_image(fig, encoded_image)
    update_image_analysis_figure_layout(fig, encoded_image.width, encoded_image.height, False)
    return fig


def add_layout_image(fig: go.Figure, encoded_image: EncodedImage):
    """Adds `encoded_image` as a layout image to `fig`.
    Args:
        fig (go.Figure): Figure to add layout image to.
        encoded_image (EncodedImage): Encoded image to add as layout image.
    """
    fig.add_layout_image(
        source=encoded_image.contents,
        xref="x",
        yref="y",
        x=0,
        y=0,
        sizex=encoded_image.width,
        sizey=encoded_image.height,
        sizing="stretch",  # Use stretch for pixel-perfect mapping
        opacity=1,
        layer="below",
    )


# Update image analysis figure layout
def update_image_analysis_figure_layout(
    fig: go.Figure, width: int, height: int, show_annotation: bool = False
) -> go.Figure:
    """Update the layout of an image analysis figure.
    Args:
        fig (go.Figure): Image analysis figure to update.
        width (int): Width of the encoded image that was added as a layout image to the figure.
        height (int): Height of the encoded image that was added as a layout image to the figure.
        show_annotation (bool, optional): Whether to show a help annotation. Defaults to False.
    Returns:
        go.Figure: The image analysis figure `fig` that was passed in.
    """
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
                "visible": show_annotation,
            }
        ]
        if show_annotation
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
