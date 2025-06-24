import argparse
from timeit import default_timer as timer
from typing import Any

import dash
import dash_bootstrap_components as dbc
import diskcache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import DiskcacheManager, dcc, html, set_props
from dash.dependencies import Input, Output, State
from dash.development.base_component import ComponentType
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
from scennia.app.image import (
    calculate_image_hash,
    concentration_color,
    create_image_analysis_figure,
    create_processed_image_analysis_figure,
    crop_cell,
    decode_image,
    encode_image,
    lactate_concentration_color,
    lactate_resistance_color,
)
from scennia.app.layout import (
    cell_info_processed_placeholder,
    create_layout,
    summary_placeholder,
)
from scennia.app.model import ModelManager

# ML model manager
MODEL_MANAGER = ModelManager()

# Stored/cached data manager
DATA_MANAGER = DataManager()

# Create dash application
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
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
    Output("image-analysis", "figure"),
    Input("show-segmentation", "value"),
    State("image-analysis", "figure"),
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

# Set layout
app.layout = create_layout()


# Process arguments
config = {"resave_processed_data": False}


def main():
    parser = argparse.ArgumentParser(description="Cell Analysis App with Lactate Classification")
    parser.add_argument("--cache_path", type=str, default="cache", help="Path to load and save cached data from")
    parser.add_argument("--model_path", type=str, default=None, help="Path to ONNX classification model")
    parser.add_argument("--lazy_load", action="store_true", help="Lazily load ONNX classification model")
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

    # Run the app
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()


# Show prepared images
@app.callback(
    Output("prepared-images", "options"),
    Input("refresh-prepared-images", "n_clicks"),
)
def show_prepared_images_callback(n_clicks):
    reload = n_clicks is not None and n_clicks > 0
    prepared_images = DATA_MANAGER.get_prepared_images(reload)
    options = []
    for i, prepared_image in enumerate(prepared_images):
        image_data = DATA_MANAGER.get_image_data(prepared_image.hash)
        if image_data is None:
            encoded_image = encode_image(prepared_image.compressed_image)
            image_data = DATA_MANAGER.update_image_data(prepared_image.hash, ImageData(encoded_image=encoded_image))
        options.append({
            "label": html.Img(src=image_data.encoded_image.contents, className="w-100 h-100 rounded"),
            "value": i,
            "input_id": f"prepared-image-input-{i}",
            "value_id": f"prepared-image-value-{i}",
        })
    set_props("prepared-image-count", {"children": f"{len(options)} Images Loaded"})
    return options


# Set header info. Shared code between `show_prepared_image` and `show_uploaded_image`.
def set_header_info(image_data: ImageData, processed_data: ProcessedData | None):
    # Get actual lacate concentration and file name if available
    actual_lactate_concentration = ""
    file_name = ""
    meta_data = image_data.meta_data
    if meta_data is not None:
        actual_lactate_concentration = f"Actual lactate concentration: {meta_data.actual_lactate_concentration}mM"
        file_name = f"File: {meta_data.file_name}"
    set_props("actual-lactate-concentration", {"children": actual_lactate_concentration})
    set_props("image-filename", {"children": file_name})

    # Process image
    if processed_data is not None:
        cell_count = f"Detected {len(processed_data.cells)} cells"
        median_cell_area = f"Median cell area: {processed_data.median_area_um:.2f}μm²"
        mean_cell_area = f"Mean cell area: {processed_data.mean_area_um:.2f}μm²"
    else:
        cell_count = ""
        median_cell_area = ""
        mean_cell_area = ""
    set_props("detected-cell-count", {"children": cell_count})
    set_props("median-cell-area", {"children": median_cell_area})
    set_props("mean-cell-area", {"children": mean_cell_area})

    # Reset cell info to placeholders
    set_props("cell-lactate-concentration", {"children": ""})
    set_props("cell-id", {"children": ""})
    set_props("cell-info", {"children": cell_info_processed_placeholder})


# Disable prepared images, upload, and switches while running.
RUNNING_DISABLE = [
    (Output("prepared-images", "disabled"), True, False),
    (Output("upload-image", "disabled"), True, False),
    (Output("show-classification", "disabled"), True, False),
    (Output("show-segmentation", "disabled"), True, False),
]


# Show prepared image callback
@app.callback(
    [
        Output("image-analysis", "figure", allow_duplicate=True),
        Output("summary", "children", allow_duplicate=True),
        Output("image-hash-store", "data", allow_duplicate=True),
    ],
    Input("prepared-images", "value"),
    State("show-segmentation", "value"),
    background=False,
    running=RUNNING_DISABLE,
    prevent_initial_call=True,
)
def show_prepared_image_callback(index, show_segmentation):
    if index is None:
        raise dash.exceptions.PreventUpdate

    prepared_image = DATA_MANAGER.get_prepared_image(index)
    if prepared_image is None:
        raise dash.exceptions.PreventUpdate

    image = prepared_image.compressed_image
    hash = prepared_image.hash
    print(f"Show prepared image: {index} with hash {hash}")

    # Encode image
    encoded_image = encode_image(image)

    # Update image data
    image_data = DATA_MANAGER.update_image_data(hash, ImageData(encoded_image=encoded_image))

    # Process image
    processed_data = get_processed_data_or_process_image(hash, image_data)
    if processed_data is not None:
        figure = create_processed_image_analysis_figure(image_data, processed_data, show_segmentation)
        summary = create_summary(image_data, processed_data)
    else:
        figure = create_image_analysis_figure(encoded_image)
        summary = summary_placeholder

    # With `--resave_processed_data`: resave processed data to save default values
    if processed_data and config["resave_processed_data"]:
        DATA_MANAGER.save_processed_data(hash, processed_data)

    # Set header info
    set_header_info(image_data, processed_data)

    return (
        figure,
        summary,
        hash,
    )


# Show uploaded image callback
@app.callback(
    [
        Output("image-analysis", "figure", allow_duplicate=True),
        Output("summary", "children", allow_duplicate=True),
        Output("image-hash-store", "data", allow_duplicate=True),
    ],
    Input("upload-image", "contents"),
    [
        State("upload-image", "filename"),
        State("show-segmentation", "value"),
    ],
    background=False,
    running=RUNNING_DISABLE,
    prevent_initial_call=True,
)
def show_uploaded_image_callback(contents, file_name, show_segmentation):
    if contents is None or file_name is None:
        raise dash.exceptions.PreventUpdate

    print(f"Show uploaded image: {file_name}")
    set_props("image-filename", {"children": file_name})

    # Get meta data from file name
    meta_data = ImageMetaData(file_name=file_name)

    # Decode uploaded image
    decode_start = timer()
    image = decode_image(contents)
    dash.callback_context.record_timing("decode", timer() - decode_start, "Decode uploaded image")

    # Hash uploaded image
    hash_start = timer()
    hash = calculate_image_hash(image)
    dash.callback_context.record_timing("hash", timer() - hash_start, "Hash uploaded image")

    # Save uploaded image
    save_uncompressed_start = timer()
    DATA_MANAGER.save_uncompressed_image(hash, meta_data.file_extension, image)
    dash.callback_context.record_timing(
        "save_uncompressed", timer() - save_uncompressed_start, "Save uncompressed image"
    )

    # Save compressed image
    save_compressed_start = timer()
    DATA_MANAGER.save_compressed_image(hash, image)
    dash.callback_context.record_timing("save_compressed", timer() - save_compressed_start, "Save compressed image")

    # Encode image
    encoded_image = encode_image(image)

    # Update and save image data
    save_data_start = timer()
    image_data = ImageData(meta_data=meta_data, encoded_image=encoded_image)
    DATA_MANAGER.save_image_data(hash, image_data)
    dash.callback_context.record_timing("save_image_data", timer() - save_data_start, "Save image data")

    # Process image
    processed_data = get_processed_data_or_process_image(hash, image_data)
    if processed_data is not None:
        figure = create_processed_image_analysis_figure(image_data, processed_data, show_segmentation)
        summary = create_summary(image_data, processed_data)
    else:
        figure = create_image_analysis_figure(encoded_image)
        summary = summary_placeholder

    # With `--resave_processed_data`: resave processed data to save default values
    if processed_data and config["resave_processed_data"]:
        DATA_MANAGER.save_processed_data(hash, processed_data)

    # Set header info
    set_header_info(image_data, processed_data)

    return (
        figure,
        summary,
        hash,
    )


# Process the uncompressed image and saves the processed data
def process_and_save_data(hash: str, image: ImageFile) -> ProcessedData:
    print(f"Processing image with hash {hash}")

    # Process image
    cellpose_model = MODEL_MANAGER.get_cellpose_model()

    processing_start = timer()

    # Convert uncompressed image to array for processing
    image_array = np.asarray(image)

    # Process with cellpose (using default values)
    flow_threshold = 0.4  # Default value
    cell_prob_threshold = 0.0  # Default value

    segmentation_start = timer()
    result = cellpose_model.eval([image_array], flow_threshold=flow_threshold, cellprob_threshold=cell_prob_threshold)
    dash.callback_context.record_timing("segmentation", timer() - segmentation_start, "Segmentation")

    mask = result[0][0]
    props = regionprops(mask)

    # Calculate aggregate data
    median_area = np.median([p.area for p in props]) if props else 0
    mean_area = np.mean([p.area for p in props]) if props else 0

    cells: dict[int, Cell] = {}  # Cells by ID
    cropped_images: dict[int, Image] = {}  # Cropped images by ID

    crop_and_classify_start = timer()
    for i, prop in enumerate(props):
        cell_id = i + 1
        is_large = prop.area > median_area

        # Create basic cell data first
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

        # Crop cell
        cropped_image = crop_cell(image, cell.bbox)
        cropped_images[cell_id] = cropped_image

        # Perform classification if model is available
        if MODEL_MANAGER.has_onnx_model_path():
            MODEL_MANAGER.load_onnx_model_if_needed()
            try:
                # Classify the cell crop
                cell.predicted_properties = MODEL_MANAGER.classify_cell_crop(cropped_image)
            except Exception as e:
                print(f"Error classifying cell {cell_id}: {e}")

        cells[cell_id] = cell

    dash.callback_context.record_timing("crop_and_classify", timer() - crop_and_classify_start, "Crop and classify")
    dash.callback_context.record_timing("total_processing", timer() - processing_start, "Total processing")

    # Encode cropped images
    encode_start = timer()
    cropped_encoded_images = {}
    for cell_id, cropped_image in cropped_images.items():
        cropped_encoded_images[cell_id] = encode_image(cropped_image)
    dash.callback_context.record_timing("encode_cropped_images", timer() - encode_start, "Encode cropped images")

    # Save processed data
    save_processed_start = timer()
    processed_data = ProcessedData(
        cropped_encoded_images=cropped_encoded_images,
        cells=cells,
        median_area=float(median_area),
        mean_area=float(mean_area),
    )
    DATA_MANAGER.save_processed_data(hash, processed_data)
    dash.callback_context.record_timing("save_processed_data", timer() - save_processed_start, "Save processed data")

    # Save cropped images
    save_cropped_start = timer()
    DATA_MANAGER.save_cropped_compressed_images(hash, cropped_images)
    dash.callback_context.record_timing(
        "save_cropped_compressed_images", timer() - save_cropped_start, "Save compressed cropped images"
    )

    return processed_data


# Gets the processed data if available in memory or disk store, or processes the image and saves the processed data.
# Returns None if the image needs to be processed but the uncompressed image or image data is not available for `hash`.
def get_processed_data_or_process_image(hash: str, image_data: ImageData) -> ProcessedData | None:
    # Get processed data from memory or disk
    processed_data_start = timer()
    processed_data = DATA_MANAGER.get_processed_data(hash)
    dash.callback_context.record_timing("get_processed_data", timer() - processed_data_start, "Get processed data")
    if processed_data is not None:
        return processed_data

    # Processed data not available, we need to process the image belonging to `hash`
    # Get image metadata
    meta_data = image_data.meta_data
    if meta_data is None:
        print("get_processed_data_or_process: need to process image, but image metadata is not available; skip")
        return None

    # Get uncompressed image
    image = DATA_MANAGER.get_uncompressed_image(hash, meta_data.file_extension)
    if image is None:
        print("process_image: need to process image, but uncompressed image is not available; skip")
        return None

    # Process image to get processed data, and also save it
    return process_and_save_data(hash, image)


# Creates a summary for given processed data
def create_summary(image_data: ImageData, processed_data: ProcessedData) -> Any:
    summary_start = timer()

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
    graph_style = {"height": "300px", "margin": "0"}
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
            fig.update_layout(
                height=300,
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                font={"size": 12},
                showlegend=False,  # Hide color legend since colors are self-explanatory
            )
            # Disable zoom
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            # Show more info on hover.
            fig.update_traces(hovertemplate="<b>%{x}mM</b><br>Count: %{y}<extra></extra>")
            # Add graph
            graph = dcc.Graph(figure=fig, config=graph_config, style=graph_style)
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
            fig.update_layout(
                height=300,
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                font={"size": 12},
                showlegend=False,  # Hide color legend since colors are self-explanatory
            )
            # Disable zoom
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            # Show counts and show more info on hover.
            fig.update_traces(
                textinfo="value+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value} (%{percent})<extra></extra>",
            )
            # Add graph
            graph = dcc.Graph(figure=fig, config=graph_config, style=graph_style)
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
    fig.update_layout(
        height=300,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        font={"size": 12},
        showlegend=False,  # Hide color legend since colors are self-explanatory
    )
    # Disable zoom
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    # Show counts and show more info on hover.
    fig.update_traces(
        hovertemplate="Cell area: %{x}μm²<extra></extra>",
    )
    # Add graph
    graph = dcc.Graph(figure=fig, config=graph_config, style=graph_style)
    graphs.append(dbc.Col(graph, width=12))

    # Add graphs to summary
    content.append(dbc.Row(graphs, className="g-2"))

    dash.callback_context.record_timing("summary", timer() - summary_start, "Create summary")

    return content


# Classification switch toggled callback callback
@app.callback(
    [
        Output("image-analysis", "figure", allow_duplicate=True),
        Output("show-segmentation", "value", allow_duplicate=True),
    ],
    Input("show-classification", "value"),
    [
        State("image-hash-store", "data"),
        State("show-segmentation", "value"),
    ],
    prevent_initial_call=True,
)
def classification_switch_toggled_callback(show_classification, hash, show_segmentation):
    show_segmentation_update = dash.no_update
    if not show_segmentation and show_classification:
        # Auto-enable show segmentation when show classification is enabled
        show_segmentation_update = True
        show_segmentation = True

    if hash is None:
        # If there is no figure yet, just update the segmentation switch
        return (dash.no_update, show_segmentation_update)

    # Get image data
    image_data = DATA_MANAGER.get_image_data(hash)
    if image_data is None:
        raise dash.exceptions.PreventUpdate

    # Get processed data
    processed_data = get_processed_data_or_process_image(hash, image_data)
    if processed_data is None:
        raise dash.exceptions.PreventUpdate

    # Create figure
    figure = create_processed_image_analysis_figure(image_data, processed_data, show_segmentation, show_classification)

    return (figure, show_segmentation_update)


# Show clicked cell callback
@app.callback(
    Output("cell-info", "children", allow_duplicate=True),
    Input("image-analysis", "clickData"),
    [
        State("image-hash-store", "data"),
    ],
    prevent_initial_call=True,
)
def show_clicked_cell_callback(click_data, hash):
    if click_data is None or hash is None:
        raise dash.exceptions.PreventUpdate

    # Get image data
    image_data_start = timer()
    image_data = DATA_MANAGER.get_image_data(hash)
    dash.callback_context.record_timing("get_image_data", timer() - image_data_start, "Get image data")
    if image_data is None:
        print("show_clicked_cell: no image data; skip")
        raise dash.exceptions.PreventUpdate
    encoded_image = image_data.encoded_image

    # Get processed data
    processed_data_start = timer()
    processed_data = DATA_MANAGER.get_processed_data(hash)
    dash.callback_context.record_timing("get_processed_data", timer() - processed_data_start, "Get processed data")
    if processed_data is None:
        print("show_clicked_cell: no processed data; skip")
        raise dash.exceptions.PreventUpdate
    cropped_encoded_images = processed_data.cropped_encoded_images
    cells = processed_data.cells

    # Find clicked cell
    find_start = timer()

    # Get click coordinates
    if "points" not in click_data or not click_data["points"] or len(click_data["points"]) == 0:
        print("show_clicked_cell: no points in click data; skip")
        raise dash.exceptions.PreventUpdate
    point = click_data["points"][0]

    # Check customdata
    if "customdata" not in point:
        # No customdata, use closest cell approach
        if "x" not in point or "y" not in point:
            print("show_clicked_cell: no x or y in point; skip")
            raise dash.exceptions.PreventUpdate

        click_x = point["x"]
        click_y = point["y"]

        # Find the closest cell to the click coordinates
        closest_cell = None
        min_distance = float("inf")

        for cell in cells.values():
            dx = cell.centroid_x - click_x
            dy = cell.centroid_y - click_y
            distance = dx * dx + dy * dy  # Squared distance is enough for comparison

            if distance < min_distance:
                min_distance = distance
                closest_cell = cell

        # Set a maximum distance threshold (radius squared)
        max_distance_threshold = 30 * 30  # 30 pixel radius
        if min_distance > max_distance_threshold:
            return html.P("Click closer to a cell center to view its details", className="text-muted")

        # We found a cell close to the click
        cell = closest_cell
    else:
        # Extract the cell ID from customdata
        customdata = point["customdata"]

        if isinstance(customdata, list):
            if len(customdata) > 0:
                # Fixed ternary operator (SIM108)
                cell_id = customdata[0][0] if isinstance(customdata[0], list) else customdata[0]
            else:
                return html.P("Invalid click data", className="text-muted")
        else:
            # Direct value
            cell_id = customdata

        # Find the cell in our data
        cell = cells[cell_id]
    dash.callback_context.record_timing("find_cell", timer() - find_start, "Find cell")

    # If no cell was found, show a placeholder.
    if not cell:
        return html.P(f"Cell data not found for ID {cell_id}", className="text-muted")

    # Get cropped image
    cropped_encoded_image = cropped_encoded_images[cell_id]

    show_cropped_start = timer()

    # Get predicted and relative lactate concentration, along with the confidence of the prediction.
    (concentration, r_concentration, confidence) = cell.lactate_concentration(image_data.actual_lactate_concentration())

    # Try to color based on relative lactate concentration
    border_color = rgb_to_hex(concentration_color(concentration, r_concentration))

    if cropped_encoded_image:
        # Use the cached cropped image directly
        cell_image = html.Img(
            src=cropped_encoded_image.contents,
            className="img-fluid",
            style={
                "width": "100%",
                "max-height": "400px",
                "object-fit": "cover",
                "border": f"3px solid {border_color}",
            },
        )
    else:
        print("Creating cropped image dynamically (fallback)")

        encoded_image = image_data.encoded_image

        # Create the cropped image dynamically (fallback)
        y0, x0, y1, x1 = cell.bbox
        padding = 10
        y0 = max(0, y0 - padding)
        x0 = max(0, x0 - padding)
        y1 = min(encoded_image.height, y1 + padding)
        x1 = min(encoded_image.width, x1 + padding)

        # Create a zoomed-in view of the cell
        cell_fig = go.Figure()

        # Add the original image
        cell_fig.add_layout_image({
            "source": encoded_image,
            "xref": "x",
            "yref": "y",
            "x": 0,
            "y": 0,
            "sizex": x1 - x0,
            "sizey": y1 - y0,
            "sizing": "stretch",
            "opacity": 1,
            "layer": "below",
        })

        # Draw a rectangle around the cell
        cell_fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line={"color": border_color, "width": 3},
            fillcolor="rgba(0,0,0,0)",
        )

        # Update layout to zoom in on the cell
        cell_fig.update_layout(
            autosize=True,
            height=300,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        # Set axes ranges to zoom on the cell
        cell_fig.update_xaxes(
            range=[x0, x1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            visible=False,
            constrain="domain",
        )
        cell_fig.update_yaxes(
            range=[y1, y0],  # Reversed for image coordinates
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
            scaleratio=1.0,
        )

        # Create graph object for cell image
        cell_image = dcc.Graph(
            figure=cell_fig, config={"displayModeBar": False}, style={"width": "100%", "height": "300px"}
        )

    dash.callback_context.record_timing("show_cropped", timer() - show_cropped_start, "Show cropped image")

    # Create cell details with predicted properties
    details_start = timer()
    cell_info: list[ComponentType] = [
        cell_image,
    ]

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
        cell_info.append(
            html.Div(
                className="p-2 my-3 border rounded border-primary-subtle bg-secondary-subtle",
                children=[
                    html.H6("Cell Classification"),
                    html.Ul(className="my-0", children=facts),
                ],
            )
        )

    # Add cell data
    cell_info.extend([
        html.H6("Cell Data"),
        html.Ul(
            className="my-0",
            children=[
                html.Li(f"Area: {cell.area_um:.2f}μm²"),
                html.Li(f"Perimeter: {cell.perimeter_um:.2f}μm"),
                html.Li(f"Eccentricity: {cell.eccentricity:.3f}"),
                html.Li(f"Centroid: ({cell.centroid_x:.1f}, {cell.centroid_y:.1f})"),
            ],
        ),
    ])
    dash.callback_context.record_timing("details", timer() - details_start, "Create details")

    set_props(
        "cell-lactate-concentration",
        {"children": f"Lactate concentration: {concentration}mM" if concentration is not None else ""},
    )
    set_props("cell-id", {"children": f"#{cell_id}" if cell_id is not None else ""})

    return cell_info


# Add callback for model status
@app.callback(
    Output("model-status", "children"),
    Input("upload-image", "id"),  # Trigger on app load by using a static component ID
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
