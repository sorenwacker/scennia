import argparse
from typing import Any

import dash
import dash_bootstrap_components as dbc
import diskcache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import DiskcacheManager, callback, dcc, html, set_props
from dash.dependencies import Input, Output, State
from dash.development.base_component import ComponentType
from dash.exceptions import PreventUpdate
from PIL.Image import Image
from PIL.ImageFile import ImageFile
from skimage.measure import find_contours, regionprops
from webcolors import rgb_to_hex

from scennia.app.data import (
    Cell,
    DataManager,
    ImageData,
    ImageMetaData,
    ProcessedData,
    confidence_into_english,
    relative_lactate_concentration_into_resistance,
)
from scennia.app.figure import (
    ImageAnalysisFilter,
    concentration_color,
    create_image_analysis_figure,
    create_placeholder_image_analysis_figure,
    lactate_concentration_color,
    lactate_resistance_color,
)
from scennia.app.image import (
    calculate_image_hash,
    crop_cell,
    decode_image,
    encode_image,
)
from scennia.app.layout import (
    CELL_INFO_BODY_ID,
    CELL_INFO_ID_ID,
    CELL_INFO_LACTATE_CONCENTRATION_ID,
    HASH_STORE,
    IMAGE_ANALYSIS_ACTUAL_LACTATE_CONCENTRATION_ID,
    IMAGE_ANALYSIS_CLASSIFICATION_ID,
    IMAGE_ANALYSIS_FILTER_STORE,
    IMAGE_ANALYSIS_GRAPH_ID,
    IMAGE_ANALYSIS_LOADING_ID,
    IMAGE_ANALYSIS_SEGMENTATION_ID,
    PREPARED_IMAGES_COUNT_ID,
    PREPARED_IMAGES_ID,
    PREPARED_IMAGES_REFRESH_ID,
    PROCESSED_HASH_STORE_ID,
    SELECTED_CELL_STORE,
    STATISTICS_BODY_ID,
    STATISTICS_CELL_AREA_ID,
    STATISTICS_CELL_COUNT_ID,
    STATISTICS_FITER_ID,
    STATISTICS_FITER_RESET_ID,
    UPLOAD_IMAGE_FILE_NAME_ID,
    UPLOAD_IMAGE_ID,
    cell_info_processed_placeholder,
    create_layout,
    statistics_placeholder,
)
from scennia.app.model import ModelManager
from scennia.app.timer import Timer

# ML model manager
MODEL_MANAGER = ModelManager()

# Stored/cached data manager
DATA_MANAGER = DataManager()

# Create dash application
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    background_callback_manager=DiskcacheManager(diskcache.Cache("./cache")),
    suppress_callback_exceptions=True,
    title="SCENNIA: Prototype Image Analysis Platform",
    update_title=None,  # type: ignore[reportArgumentType]
)

# Define the client-side callback for toggling annotations
app.clientside_callback(
    """
    function(showSegmentation, figure) {
        if (!figure || !figure.data) {
            return window.dash_clientside.no_update;
        }

        // Create a copy of the figure to avoid modifying the original
        const newFigure = JSON.parse(JSON.stringify(figure));

        // Toggle traces visibility
        for (let i = 0; i < newFigure.data.length; i++) {
            newFigure.data[i].visible = showSegmentation;
        }

        // Toggle annotation visibility
        if (newFigure.layout && newFigure.layout.annotations) {
            for (let i = 0; i < newFigure.layout.annotations.length; i++) {
                newFigure.layout.annotations[i].visible = showSegmentation;
            }
        }

        return newFigure;
    }
    """,
    Output(IMAGE_ANALYSIS_GRAPH_ID, "figure", allow_duplicate=True),
    Input(IMAGE_ANALYSIS_SEGMENTATION_ID, "value"),
    State(IMAGE_ANALYSIS_GRAPH_ID, "figure"),
    prevent_initial_call=True,
)

# Add inline CSS
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        /* Add radio-group. Source: https://www.dash-bootstrap-components.com/docs/components/button_group/ */
        .radio-group .form-check {
            padding-left: 0;
        }

        .radio-group .btn-group > .form-check:not(:last-child) > .btn {
            border-top-right-radius: 0;
            border-bottom-right-radius: 0;
        }

        .radio-group .btn-group > .form-check:not(:first-child) > .btn {
            border-top-left-radius: 0;
            border-bottom-left-radius: 0;
            margin-left: -1px;
        }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Process arguments
config = {"resave_processed_data": False}


def main():
    parser = argparse.ArgumentParser(description="Cell Analysis App with Lactate Classification")
    parser.add_argument("--cache_path", type=str, default="cache", help="Path to load and save cached data from")
    parser.add_argument("--model_path", type=str, default=None, help="Path to ONNX classification model")
    parser.add_argument("--lazy_load", action="store_true", help="Lazily load ONNX classification model")
    parser.add_argument(
        "--hide_image_upload",
        action="store_true",
        help="Hide the image uploader",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--resave_processed_data",
        action="store_true",
        help="Save processed data after loading it, saving default values",
    )

    args = parser.parse_args()

    # Set cache path
    if args.cache_path:
        DATA_MANAGER.set_cache_path(args.cache_path)

    # Set model path if provided
    if args.model_path:
        MODEL_MANAGER.set_onnx_model_path(args.model_path)

    # Load classification model eagerly
    if args.model_path and not args.lazy_load:
        success = MODEL_MANAGER.load_onnx_model_if_needed()
        if success:
            print(f"Successfully loaded classification model from {args.model_path}")
        else:
            print(f"Failed to load classification model from {args.model_path}")
            print("App will run with basic size-based classification only")
    elif not args.lazy_load:
        print("No classification model provided. App will run with basic size-based classification only")

    # Custom config
    config["resave_processed_data"] = args.resave_processed_data is True
    show_image_upload = args.hide_image_upload is not True

    # Set layout
    app.layout = create_layout(show_image_upload)

    # Run the app
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()


@callback(
    Output(PREPARED_IMAGES_COUNT_ID, "children"),
    Output(PREPARED_IMAGES_ID, "options"),
    Input(PREPARED_IMAGES_REFRESH_ID, "n_clicks"),
)
def prepared_images_update_callback(refresh_clicks: int | None) -> tuple[str, list]:
    """Prepared images update callback.
    Args:
        n_clicks (int | None): Number of times the "Refresh Images" button was clicked.
    Returns:
        tuple[str, list]: Number of images loaded and a list of options for `bc.RadioItems`.
    """
    reload = refresh_clicks is not None and refresh_clicks > 0

    print(f"Update prepared images: reload={reload}")

    with Timer("get_prepared_images", "Get prepared images"):
        prepared_images = DATA_MANAGER.get_prepared_images(reload)

    with Timer("update_prepared_images", "Update prepared images"):
        options = []
        for i, prepared_image in enumerate(prepared_images):
            image_data = DATA_MANAGER.get_image_data(prepared_image.hash)
            if image_data is None:
                # Encode the image and update image data, ensuring image data is available.
                encoded_image = encode_image(prepared_image.compressed_image)
                image_data = DATA_MANAGER.update_image_data(prepared_image.hash, ImageData(encoded_image=encoded_image))
            options.append({
                "label": html.Img(src=image_data.encoded_image.contents, className="w-100 h-100 rounded"),
                "value": i,
                "input_id": f"prepared-image-input-{i}",
                "value_id": f"prepared-image-value-{i}",
            })

    return f"{len(options)} Images Loaded", options


def image_analysis_running_components(final_callback: bool) -> list[tuple[Output, Any, Any]]:
    """Returns the list of components to change while an image analysis callback is running.
    Args:
        last_step (bool): Whether we are in the final callback of a callback chain, which will enable the components
        again when the callback is done running.
    Returns:
        list[tuple[Output, Any, Any]]: List of components to change
    """
    return [
        (Output(PREPARED_IMAGES_ID, "disabled"), True, not final_callback),
        (Output(UPLOAD_IMAGE_ID, "disabled"), True, not final_callback),
        (Output(IMAGE_ANALYSIS_SEGMENTATION_ID, "disabled"), True, not final_callback),
        (Output(IMAGE_ANALYSIS_CLASSIFICATION_ID, "disabled"), True, not final_callback),
        (Output(IMAGE_ANALYSIS_LOADING_ID, "display"), "show", "auto" if final_callback else "show"),
    ]


# Click prepared image callback.
@callback(
    Output(HASH_STORE, "data", allow_duplicate=True),
    Output(IMAGE_ANALYSIS_FILTER_STORE, "data", allow_duplicate=True),
    Output(SELECTED_CELL_STORE, "data", allow_duplicate=True),
    Input(PREPARED_IMAGES_ID, "value"),
    running=image_analysis_running_components(False),
    prevent_initial_call=True,
)
def click_prepared_image_callback(index: int | None) -> tuple[str, None, None]:
    if index is None:
        raise PreventUpdate

    prepared_image = DATA_MANAGER.get_prepared_image(index)
    if prepared_image is None:
        raise PreventUpdate

    image = prepared_image.compressed_image
    hash = prepared_image.hash
    print(f"Clicked prepared image: {index} with hash {hash}")

    # Update image data, ensuring the encoded image is available.
    with Timer("encode_prepared_image", "Encode prepared image"):
        encoded_image = encode_image(image)
    with Timer("update_image_data", "Update image data"):
        DATA_MANAGER.update_image_data(hash, ImageData(encoded_image=encoded_image))

    # Return `hash` for `HASH_STORE`, reset `IMAGE_ANALYSIS_FILTER_STORE` and `SELECTED_CELL_STORE`.
    return hash, None, None


# Upload image callback.
@callback(
    Output(HASH_STORE, "data", allow_duplicate=True),
    Output(IMAGE_ANALYSIS_FILTER_STORE, "data", allow_duplicate=True),
    Output(SELECTED_CELL_STORE, "data", allow_duplicate=True),
    Input(UPLOAD_IMAGE_ID, "contents"),
    State(UPLOAD_IMAGE_ID, "filename"),
    running=image_analysis_running_components(False),
    prevent_initial_call=True,
)
def upload_image_callback(contents: str | None, file_name: str | None) -> tuple[str, None, None]:
    if contents is None or file_name is None:
        raise PreventUpdate

    print(f"Uploaded image: {file_name}")

    # Update header
    set_props(UPLOAD_IMAGE_FILE_NAME_ID, {"children": file_name})

    # Create image meta data from uploaded file name.
    meta_data = ImageMetaData(file_name=file_name)

    with Timer("decode", "Decode uploaded image"):
        image = decode_image(contents)
    with Timer("hash", "Hash uploaded image"):
        hash = calculate_image_hash(image)
    with Timer("save_uncompressed", "Save uncompressed image"):
        DATA_MANAGER.save_uncompressed_image(hash, meta_data.file_extension, image)
    with Timer("save_compressed", "Save compressed image"):
        DATA_MANAGER.save_compressed_image(hash, image)
    with Timer("encode", "Encode uncompressed image"):
        encoded_image = encode_image(image)
    with Timer("save_image_data", "Save image data"):
        image_data = ImageData(meta_data=meta_data, encoded_image=encoded_image)
        DATA_MANAGER.save_image_data(hash, image_data)

    # Return `hash` for `HASH_STORE`, reset `IMAGE_ANALYSIS_FILTER_STORE` and `SELECTED_CELL_STORE`.
    return hash, None, None


# Image hash change callback: ensures the image is processed, and sets `PROCESSED_HASH_STORE_ID`.
@callback(
    Output(PROCESSED_HASH_STORE_ID, "data"),
    Input(HASH_STORE, "data"),
    running=image_analysis_running_components(False),
    prevent_initial_call=True,
)
def image_hash_change_callback(hash: str | None) -> str:
    if hash is None:
        raise PreventUpdate
    print(f"Image hash update: {hash}")

    with Timer("get_image_data", "Get image data"):
        image_data = DATA_MANAGER.get_image_data_or_raise(hash)

    with Timer("get_processed_data", "Get processed data"):
        processed_data = DATA_MANAGER.get_processed_data(hash)
    if processed_data is None:  # Processed data not available, we need to process the image belonging to `hash`.
        # Get required data.
        if image_data.meta_data is None:
            print("image_hash_change_callback: skipping; no image meta data")
            raise PreventUpdate
        image = DATA_MANAGER.get_uncompressed_image(hash, image_data.meta_data.file_extension)
        if image is None:
            print("image_hash_change_callback: skipping; no uncompressed image")
            raise PreventUpdate
        # Process image to get processed data, and also save it.
        processed_data = process_and_save_data(hash, image)

    # With `--resave_processed_data`: resave processed data to save default values.
    if config["resave_processed_data"]:
        with Timer("resave_processed_data", "Resave processed data"):
            DATA_MANAGER.save_processed_data(hash, processed_data)

    # Update headers
    if image_data.meta_data is not None:
        actual_lactate_concentration = (
            f"Actual lactate concentration: {image_data.meta_data.actual_lactate_concentration}mM"
            if image_data.meta_data.actual_lactate_concentration is not None
            else ""
        )
        file_name = f"File: {image_data.meta_data.file_name}"
    else:
        actual_lactate_concentration = ""
        file_name = ""
    set_props(IMAGE_ANALYSIS_ACTUAL_LACTATE_CONCENTRATION_ID, {"children": actual_lactate_concentration})
    set_props(UPLOAD_IMAGE_FILE_NAME_ID, {"children": file_name})

    # Return `hash` for `PROCESSED_HASH_STORE_ID`.
    return hash


def process_and_save_data(hash: str, image: ImageFile) -> ProcessedData:
    """Process the uncompressed `image` and save the processed data.
    Args:
        hash (str): Hash of the `image`.
        image (ImageFile): Uncompressed image to process.
    Returns:
        ProcessedData: Processed data.
    """
    print(f"Processing image with hash: {hash}")

    # Load cellpose model
    cellpose_model = MODEL_MANAGER.get_cellpose_model()

    # Start processing
    processing_timer = Timer("total_processing", "Total processing")

    # Process with cellpose (using default values)
    flow_threshold = 0.4  # Default value
    cell_prob_threshold = 0.0  # Default value
    with Timer("segmentation", "Segmentation"):
        result = cellpose_model.eval(
            [np.asarray(image)], flow_threshold=flow_threshold, cellprob_threshold=cell_prob_threshold
        )
    mask = result[0][0]
    props = regionprops(mask)

    # Calculate aggregate data
    median_area = np.median([p.area for p in props]) if props else 0
    mean_area = np.mean([p.area for p in props]) if props else 0

    cells: dict[int, Cell] = {}  # Cells by cell ID
    cropped_images: dict[int, Image] = {}  # Cropped images by cell ID

    # Crop and classify
    crop_and_classify_timer = Timer("crop_and_classify", "Crop and classify")
    for i, prop in enumerate(props):
        cell_id = i + 1
        is_large = prop.area > median_area

        # Create basic cell data
        cell = Cell(
            id=cell_id,
            centroid_y=float(prop.centroid[0]),
            centroid_x=float(prop.centroid[1]),
            area=float(prop.area),
            perimeter=float(prop.perimeter),
            eccentricity=float(prop.eccentricity),
            bbox=[int(x) for x in prop.bbox],
            contour=find_contours(mask == cell_id)[0].T.tolist(),
            is_large=bool(is_large),
        )

        # Crop cell image
        cropped_image = crop_cell(image, cell.bbox)
        cropped_images[cell_id] = cropped_image

        # Perform classification if model is available.
        if MODEL_MANAGER.has_onnx_model_path():
            MODEL_MANAGER.load_onnx_model_if_needed()
            try:
                # Classify the cell, setting the predicted properties.
                cell.predicted_properties = MODEL_MANAGER.classify_cell_crop(cropped_image)
            except Exception as e:
                print(f"Error classifying cell {cell_id}: {e}")

        cells[cell_id] = cell

    crop_and_classify_timer.record()
    processing_timer.record()

    with Timer("encode_cropped_images", "Encode cropped images"):
        cropped_encoded_images = {}
        for cell_id, cropped_image in cropped_images.items():
            cropped_encoded_images[cell_id] = encode_image(cropped_image)

    with Timer("save_processed_data", "Save processed data"):
        processed_data = ProcessedData(
            cropped_encoded_images=cropped_encoded_images,
            cells=cells,
            median_area=float(median_area),
            mean_area=float(mean_area),
        )
        DATA_MANAGER.save_processed_data(hash, processed_data)

    with Timer("save_cropped_compressed_images", "Save compressed cropped images"):
        DATA_MANAGER.save_cropped_compressed_images(hash, cropped_images)

    return processed_data


# Image analysis update callback
@callback(
    Output(IMAGE_ANALYSIS_GRAPH_ID, "figure", allow_duplicate=True),
    Input(PROCESSED_HASH_STORE_ID, "data"),
    # State instead of input, because it is handled by a client-side callback.
    State(IMAGE_ANALYSIS_SEGMENTATION_ID, "value"),
    Input(IMAGE_ANALYSIS_CLASSIFICATION_ID, "value"),
    Input(IMAGE_ANALYSIS_FILTER_STORE, "data"),
    running=image_analysis_running_components(True),
    prevent_initial_call=True,
)
def image_analysis_update_callback(
    hash: str | None,
    show_segmentation: bool | None,
    show_classification: bool | None,
    filter_str: str | None,
) -> go.Figure:
    if hash is None:
        raise PreventUpdate

    with Timer("get_image_data", "Get image data"):
        image_data = DATA_MANAGER.get_image_data_or_raise(hash)
    with Timer("processed_data", "Get processed data"):
        processed_data = DATA_MANAGER.get_processed_data_or_raise(hash)

    # Normalize inputs
    show_segmentation = show_segmentation is True
    show_classification = show_classification is True
    filter = ImageAnalysisFilter() if filter_str is None else ImageAnalysisFilter.from_json(filter_str)
    print(f"Image analysis update with filter: {filter}")

    if processed_data is not None:
        with Timer("create_image_analysis_figure", "Create image analysis figure"):
            figure = create_image_analysis_figure(
                image_data, processed_data, filter, show_segmentation, show_classification
            )
    else:
        figure = create_placeholder_image_analysis_figure(image_data.encoded_image)

    return figure


# Statistics update callback
@callback(
    Output(STATISTICS_BODY_ID, "children"),
    Input(PROCESSED_HASH_STORE_ID, "data"),
    prevent_initial_call=True,
)
def statistics_update_callback(hash: str | None):
    if hash is None:
        raise PreventUpdate

    with Timer("get_image_data", "Get image data"):
        image_data = DATA_MANAGER.get_image_data_or_raise(hash)
    with Timer("processed_data", "Get processed data"):
        processed_data = DATA_MANAGER.get_processed_data_or_raise(hash)

    with Timer("create_statistics", "Create statistics"):
        body = create_statistics(image_data, processed_data) if processed_data is not None else statistics_placeholder

    # Update headers
    set_props(STATISTICS_CELL_COUNT_ID, {"children": f"{len(processed_data.cells)} cells"})
    median = f"median: {processed_data.median_area_um:.2f}μm²"
    mean = f"mean: {processed_data.mean_area_um:.2f}μm²"
    set_props(STATISTICS_CELL_AREA_ID, {"children": f"Cell area: {median}, {mean}"})

    return body


def create_statistics(image_data: ImageData, processed_data: ProcessedData) -> list[ComponentType]:
    """Creates the statistics body for given processed data.
    Args:
        image_data (ImageData): Image data to use in creating statistics.
        processed_data (ProcessedData): Processed data to create statistics for.
    Returns:
        list[ComponentType]: List of Dash components.
    """

    content: list[ComponentType] = []

    # Gather concentration data
    actual_lactate_concentration = image_data.actual_lactate_concentration()
    concentration_counts = {}
    lactate_resistance_counts = {}
    classification_available = False
    for cell in processed_data.cells.values():
        concentration, lactate_resistance, _ = cell.lactate_concentration(actual_lactate_concentration)
        if concentration is not None:
            classification_available = True
            concentration_counts[concentration] = concentration_counts.get(concentration, 0) + 1
        if lactate_resistance is not None:
            classification_available = True
            lactate_resistance = relative_lactate_concentration_into_resistance(lactate_resistance)
            lactate_resistance_counts[lactate_resistance] = lactate_resistance_counts.get(lactate_resistance, 0) + 1

    # Default graph config and styles
    graph_config: dcc.Graph.Config = {"displayModeBar": False}
    graph_style = {"height": "250px", "margin": "0"}
    graph_margin = {"l": 0, "r": 0, "t": 30, "b": 0}
    graph_font = {"size": 12}
    graphs = []

    # Add lactate concentration graphs
    if classification_available:
        # Create cell concentration bar plot
        if concentration_counts:
            all_concentrations = [0, 5, 10, 20, 40, 80]  # Ensure all concentration levels are represented
            plot_data = []
            for concentration in all_concentrations:
                count = concentration_counts.get(concentration, 0)
                color = lactate_concentration_color(concentration)
                color_label = px.colors.label_rgb(color)
                plot_data.append({
                    "Concentration": f"{concentration}",  # Convert to string to use discrete values on x axis
                    "Count": count,
                    "Color": color_label,
                })
            df = pd.DataFrame(plot_data)

            # Create bar plot
            fig = px.bar(
                df,
                x="Concentration",
                y="Count",
                title="Cells by Lactate Concentration",
                labels={"Concentration": "Lactate Concentration [mM]", "Count": "Number of Cells"},
                text="Count",
                text_auto=True,
                color="Color",
                color_discrete_map={row["Color"]: row["Color"] for _, row in df.iterrows()},
            )
            # Update layout
            fig.update_layout(margin=graph_margin, font=graph_font, showlegend=False)
            # Disable zoom
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            # Show more info on hover
            fig.update_traces(hovertemplate="<b>%{x}mM</b><br>Count: %{y}<extra></extra>")
            # Add graph
            graph = dcc.Graph(id="cell-concentration-graph", figure=fig, config=graph_config, style=graph_style)
            graphs.append(dbc.Col(graph, width=8))

        # Create lactate resistance pie chart
        if lactate_resistance_counts:
            plot_data = []
            for lactate_resistance, count in sorted(lactate_resistance_counts.items()):
                color = lactate_resistance_color(lactate_resistance)
                color_label = px.colors.label_rgb(color)
                plot_data.append({
                    "Lactate Resistance": lactate_resistance,
                    "Count": count,
                    "Color": color_label,
                })
            df = pd.DataFrame(plot_data)

            # Create lactate resistance pie chart
            fig = px.pie(
                df,
                names="Lactate Resistance",
                values="Count",
                title="Cells by Lactate Resistance",
                labels={"Lactate Resistance": "Lactate Resistance", "Count": "Number of Cells"},
                color="Color",
                color_discrete_map={row["Color"]: row["Color"] for _, row in df.iterrows()},
            )
            # Update layout
            fig.update_layout(margin=graph_margin, font=graph_font, showlegend=False)
            # Disable zoom
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            # Show counts and show more info on hover
            fig.update_traces(
                textinfo="value+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value} (%{percent})<extra></extra>",
            )
            # Add graph
            graph = dcc.Graph(id="cell-lactate-resistance-graph", figure=fig, config=graph_config, style=graph_style)
            graphs.append(dbc.Col(graph, width=4))
    else:
        large_cells = sum(cell.is_large for cell in processed_data.cells.values())
        small_cells = len(processed_data.cells) - large_cells
        content.append(
            html.P([
                "Cell count by size: ",
                html.Span(f"{large_cells} large", style={"color": "green", "fontWeight": "bold"}),
                ", ",
                html.Span(f"{small_cells} small", style={"color": "red", "fontWeight": "bold"}),
            ])
        )

    # Create cell area violin plot
    df = pd.DataFrame([{"Area": c.area_um} for c in processed_data.cells.values()])
    fig = px.violin(
        df,
        x="Area",
        orientation="h",
        box=True,
        points="all",
        title="Cell Area Distribution",
        labels={"Area": "Cell Area [μm²]"},
    )
    # Update layout
    fig.update_layout(margin=graph_margin, font=graph_font, showlegend=False, dragmode="select")
    # Disable zoom
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    # Show cell area and kde on hover, and hide verbose violins hovers
    fig.update_traces(
        hoveron="points+kde",
        hovertemplate="Cell area: %{x}μm²<extra></extra>",
    )
    # Add graph
    graph = dcc.Graph(id="cell-area-graph", figure=fig, config=graph_config, style=graph_style)
    graphs.append(dbc.Col(graph, width=12))

    # Add graphs to summary
    content.append(dbc.Row(graphs, className="g-2"))
    return content


# Cell clicked callback: finds the ID of the clicked cell and sets `SELECTED_CELL_STORE` to it.
@callback(
    Output(SELECTED_CELL_STORE, "data", allow_duplicate=True),
    Input(IMAGE_ANALYSIS_GRAPH_ID, "clickData"),
    State(PROCESSED_HASH_STORE_ID, "data"),
    prevent_initial_call=True,
)
def cell_clicked_callback(click_data, hash: str | None) -> int:
    if click_data is None or hash is None:
        raise PreventUpdate

    # Get click coordinates
    if "points" not in click_data or not click_data["points"] or len(click_data["points"]) == 0:
        print(f"cell_clicked_callback: skipping; no points in click data: {click_data}")
        raise PreventUpdate
    point = click_data["points"][0]

    # Get custom data
    if "customdata" not in point:
        print(f"cell_clicked_callback: skipping; no 'customdata' property in clicked point: {point}")
        raise PreventUpdate
    customdata = point["customdata"]

    # Convert custom data to cell ID.
    if isinstance(customdata, list):
        if len(customdata) > 0:
            cell_id = customdata[0][0] if isinstance(customdata[0], list) else customdata[0]
        else:
            print("cell_clicked_callback: skipping; 'customdata' is an empty list")
            raise PreventUpdate
    else:
        cell_id = customdata
    print(f"Clicked cell: {cell_id}")

    return int(cell_id)


# Cell info update callback
@callback(
    Output(CELL_INFO_ID_ID, "children"),
    Output(CELL_INFO_LACTATE_CONCENTRATION_ID, "children"),
    Output(CELL_INFO_BODY_ID, "children"),
    Input(SELECTED_CELL_STORE, "data"),
    State(PROCESSED_HASH_STORE_ID, "data"),
    prevent_initial_call=True,
)
def cell_info_update_callback(selected_cell: int | None, hash: str | None):
    if selected_cell is None:
        return "", "", cell_info_processed_placeholder

    if hash is None:
        raise PreventUpdate

    # Get required data
    image_data = DATA_MANAGER.get_image_data_or_raise(hash)
    processed_data = DATA_MANAGER.get_processed_data_or_raise(hash)
    cropped_encoded_images = processed_data.cropped_encoded_images
    cells = processed_data.cells

    print(f"Cell info update: {selected_cell}")

    # Check if cell exist
    if selected_cell not in cells:
        return "", "", html.P(f"Cell '{selected_cell}' not found", className="text-muted")
    cell = cells[selected_cell]

    # Gather cell info
    content: list[ComponentType] = []

    # Get predicted and relative lactate concentration, along with the confidence of the prediction.
    (concentration, r_concentration, confidence) = cell.lactate_concentration(image_data.actual_lactate_concentration())

    # Add cropped cell image
    with Timer("add_cropped_cell_image", "Add cropped cell image"):
        cropped_encoded_image = cropped_encoded_images.get(selected_cell)
        if cropped_encoded_image is not None:
            # Try to color based on relative lactate concentration
            border_color = rgb_to_hex(concentration_color(concentration, r_concentration))
            content.append(
                html.Img(
                    src=cropped_encoded_image.contents,
                    className="img-fluid",
                    style={
                        "width": "100%",
                        "height": "375px",
                        "object-fit": "cover",
                        "border": f"3px solid {border_color}",
                    },
                )
            )

    # Add cell info
    timer = Timer("add_info", "Add info")

    # Add classification if available
    if cell.predicted_properties is not None:
        facts = []
        if concentration is not None:
            facts.append(html.Li(f"Lactate concentration: {concentration}mM"))
        if r_concentration is not None:
            facts.append(html.Li(f"Relative lactate concentration: {r_concentration}mM"))
        if confidence is not None:
            facts.append(html.Li(f"Confidence: {confidence_into_english(confidence)}"))
        if r_concentration is not None:
            facts.append(
                html.Li(
                    f"Conclusion: {relative_lactate_concentration_into_resistance(r_concentration)}",
                    className="fw-bold",
                )
            )
        content.append(
            html.Div(
                className="p-2 my-3 border rounded border-primary-subtle bg-secondary-subtle",
                children=[
                    html.H6("Cell Classification"),
                    html.Ul(className="my-0", children=facts),
                ],
            )
        )

    # Add data
    content.extend([
        html.H6("Cell Data"),
        html.Ul(
            className="my-0",
            children=[
                html.Li(f"Area: {cell.area_um:.2f}μm²"),
                html.Li(f"Perimeter: {cell.perimeter_um:.2f}μm"),
                html.Li(f"Eccentricity: {cell.eccentricity:.3f}"),
            ],
        ),
    ])

    timer.record()

    concentration_str = f"Lactate concentration: {concentration}mM" if concentration is not None else ""
    return concentration_str, f"#{selected_cell}", content


# Filter by concentration callback
@callback(
    Output(IMAGE_ANALYSIS_FILTER_STORE, "data", allow_duplicate=True),
    Input("cell-concentration-graph", "clickData"),
    State(IMAGE_ANALYSIS_FILTER_STORE, "data"),
    prevent_initial_call=True,
)
def filter_by_concentration_callback(click_data: Any | None, filter_str: str | None) -> str:
    # Get lactate concentration to filter by
    concentration = None
    if click_data is not None:
        # Get click coordinates
        if "points" not in click_data or not click_data["points"] or len(click_data["points"]) == 0:
            print(f"filter_by_concentration_callback: skipping; no points in click data: {click_data}")
            raise PreventUpdate
        point = click_data["points"][0]

        # Get concentration to filter by
        if "x" not in point:
            print(f"filter_by_concentration_callback: skipping; no 'x' property in clicked point: {point}")
            raise PreventUpdate
        concentration = int(point["x"])

    # Get filter
    filter = ImageAnalysisFilter() if filter_str is None else ImageAnalysisFilter.from_json(filter_str)

    # If previous and new filter is the same, prevent update to prevent cyclic callbacks.
    if filter.concentration == concentration:
        raise PreventUpdate

    # Update filter
    print(f"Filter by lactate concentration: {concentration}")
    filter.concentration = concentration
    return filter.to_json()


# Filter by lactate resistance callback
@callback(
    Output(IMAGE_ANALYSIS_FILTER_STORE, "data", allow_duplicate=True),
    Input("cell-lactate-resistance-graph", "clickData"),
    State(IMAGE_ANALYSIS_FILTER_STORE, "data"),
    prevent_initial_call=True,
)
def filter_by_lactate_resistance_callback(click_data: Any | None, filter_str: str | None) -> str:
    # Get lactate resistance to filter by
    resistance = None
    if click_data is not None:
        # Get click coordinates
        if "points" not in click_data or not click_data["points"] or len(click_data["points"]) == 0:
            print(f"filter_by_lactate_resistance_callback: skipping; no points in click data: {click_data}")
            raise PreventUpdate
        point = click_data["points"][0]

        # Get lactate resistance to filter by
        if "label" not in point:
            print(f"filter_by_lactate_resistance_callback: skipping; no 'label' property in clicked point: {point}")
            raise PreventUpdate
        resistance = point["label"]

    # Get filter
    filter = ImageAnalysisFilter() if filter_str is None else ImageAnalysisFilter.from_json(filter_str)

    # If previous and new filter is the same, prevent update to prevent cyclic callbacks.
    if filter.resistance == resistance:
        raise PreventUpdate

    # Update filter
    print(f"Filter by lactate resistance: {resistance}")
    filter.resistance = resistance
    return filter.to_json()


# Filter by area callback
@callback(
    Output(IMAGE_ANALYSIS_FILTER_STORE, "data", allow_duplicate=True),
    Input("cell-area-graph", "selectedData"),
    State(IMAGE_ANALYSIS_FILTER_STORE, "data"),
    prevent_initial_call=True,
)
def filter_by_area_callback(selected_data: Any | None, filter_str: str | None) -> str:
    # Get area to filter by
    area = None
    if selected_data is not None:
        # Get min and max area to filter by
        if "range" not in selected_data or "x" not in selected_data["range"]:
            print(f"filter_by_area_callback: skipping; no range['x'] in selected data: {selected_data}")
            raise PreventUpdate
        min_area, max_area = selected_data["range"]["x"]
        min_area = max(min_area, 0.0)  # Clamp to positive.
        area = (min_area, max_area)

    # Get filter
    filter = ImageAnalysisFilter() if filter_str is None else ImageAnalysisFilter.from_json(filter_str)

    # If previous and new filter is the same, prevent update to prevent cyclic callbacks.
    if filter.area == area:
        raise PreventUpdate

    # Update filter
    print(f"Filter by area: {area}")
    filter.area = area
    return filter.to_json()


# Filter update callback
@callback(
    Output(STATISTICS_FITER_ID, "children", allow_duplicate=True),
    Output(STATISTICS_FITER_RESET_ID, "disabled", allow_duplicate=True),
    Input(IMAGE_ANALYSIS_FILTER_STORE, "data"),
    prevent_initial_call=True,
)
def filter_update_callback(filter_str: str | None) -> tuple[str, bool]:
    filter = ImageAnalysisFilter() if filter_str is None else ImageAnalysisFilter.from_json(filter_str)

    facets_text = ""
    reset_disabled = True
    if filter.is_filtering():
        facets = filter.to_facet_strs()
        facets_text = ", ".join(facets)
        reset_disabled = False

    # Set filter facets text and whether filter reset button is disabled
    print(f"Update filter: facets_text={facets_text}, reset_disabled={reset_disabled}")
    return facets_text, reset_disabled


# Reset filter callback
@callback(
    Output(STATISTICS_FITER_ID, "children", allow_duplicate=True),
    Output(STATISTICS_FITER_RESET_ID, "disabled", allow_duplicate=True),
    Output(IMAGE_ANALYSIS_FILTER_STORE, "data", allow_duplicate=True),
    Input(STATISTICS_FITER_RESET_ID, "n_clicks"),
    prevent_initial_call=True,
)
def reset_filter_callback(n_clicks: int | None) -> tuple[str, bool, None]:
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    print("Reset filter")

    # Reset clickData
    set_props("cell-concentration-graph", {"clickData": None})
    set_props("cell-lactate-resistance-graph", {"clickData": None})

    # Reset selectedData
    set_props("cell-area-graph", {"selectedData": None})

    # Reset filter facets text, disable reset button, set image analysis filter to `None`.
    return "", True, None


# Add callback for model status
@callback(
    Output("model-status", "children"),
    Input(UPLOAD_IMAGE_ID, "id"),  # Trigger on app load by using a static component ID
)
def show_model_status_callback(_):
    if MODEL_MANAGER.is_onnx_model_loaded() and MODEL_MANAGER.onnx_model_metadata is not None:
        return dbc.Alert(
            [
                html.Strong("Classification Model Loaded: "),
                f"{MODEL_MANAGER.onnx_model_metadata.get('model_name', 'Unknown')}",
                html.Br(),
                html.Small("Segmentation model: cellpose_3.0."),
            ],
            color="success",
            className="mb-2",
        )
    if MODEL_MANAGER.has_onnx_model_path():
        return dbc.Alert(
            [
                html.Strong("Classification Model Not Yet Loaded"),
                html.Br(),
                html.Small("Segmentation model: cellpose_3.0. Classification model will be loaded when required."),
            ],
            color="secondary",
            className="mb-2",
        )
    return dbc.Alert(
        [
            html.Strong("No Classification Model Loaded"),
            html.Br(),
            html.Small(
                "Segmentation model: cellpose_3.0. Cell classification will use basic size-based predictions only."
            ),
        ],
        color="warning",
        className="mb-2",
    )
