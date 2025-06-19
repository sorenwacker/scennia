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

from scennia.app.data import (
    AggregateData,
    Cell,
    DataManager,
    ImageData,
    ImageMetaData,
    ProcessedData,
)
from scennia.app.image import (
    calculate_image_hash,
    create_image_analysis_figure,
    create_processed_image_analysis_figure,
    crop_cell,
    decode_image,
    encode_image,
    get_concentration_darker_color,
)
from scennia.app.layout import (
    cell_info_processed_placeholder,
    create_layout,
    summary_placeholder,
)
from scennia.app.model import model_manager

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


# Add callback for model status
@app.callback(
    Output("model-status", "children"),
    Input("upload-image", "id"),  # Trigger on app load by using a static component ID
)
def display_model_status(_):
    if model_manager.is_onnx_model_loaded() and model_manager.onnx_model_metadata is not None:
        return dbc.Alert(
            [
                html.Strong("Classification Model Loaded: "),
                f"{model_manager.onnx_model_metadata.get('model_name', 'Unknown')} with {model_manager.onnx_model_metadata.get('num_classes', 0)} classes",  # noqa: E501
                html.Br(),
                html.Small(f"Classes: {', '.join(model_manager.onnx_model_metadata.get('class_names', []))}"),
            ],
            color="success",
            className="mb-2",
        )
    return dbc.Alert(
        [
            html.Strong("No Classification Model Loaded"),
            html.Br(),
            html.Small("Cell classification will use basic size-based predictions only"),
        ],
        color="warning",
        className="mb-2",
    )


# Show prepared images
@app.callback(
    Output("prepared-images", "options"),
    Input("upload-image", "id"),  # Trigger on app load by using a static component ID
)
def show_prepared_images(_):
    prepared_images = DATA_MANAGER.get_prepared_images()
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
    return options


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
    running=[  # Disable prepared images, upload, and process while running.
        (Output("prepared-images", "disabled"), True, False),
        (Output("upload-image", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def show_prepared_image(index, show_segmentation):
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
    processed_data = get_processed_data_or_process_image(hash, image_data)
    if processed_data is not None:
        figure = create_processed_image_analysis_figure(image_data, processed_data, show_segmentation)
        cell_count = f"Detected {len(processed_data.cells)} cells"
        summary = create_summary(processed_data)
    else:
        figure = create_image_analysis_figure(encoded_image)
        cell_count = ""
        summary = summary_placeholder
    set_props("detected-cell-count", {"children": cell_count})

    # Reset cell info to placeholders
    set_props("cell-lactate-concentration", {"children": ""})
    set_props("cell-id", {"children": ""})
    set_props("cell-info", {"children": cell_info_processed_placeholder})

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
    running=[  # Disable prepared images, upload, and process while running.
        (Output("prepared-images", "disabled"), True, False),
        (Output("upload-image", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def show_uploaded_image(contents, file_name, show_segmentation):
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

    # Get actual lacate concentration if available
    actual_lactate_concentration = ""
    if meta_data.actual_lactate_concentration is not None:
        actual_lactate_concentration = f"Actual lactate concentration: {meta_data.actual_lactate_concentration}mM"
    set_props("actual-lactate-concentration", {"children": actual_lactate_concentration})

    # Process image
    processed_data = get_processed_data_or_process_image(hash, image_data)
    if processed_data is not None:
        figure = create_processed_image_analysis_figure(image_data, processed_data, show_segmentation)
        cell_count = f"Detected {len(processed_data.cells)} cells"
        summary = create_summary(processed_data)
    else:
        figure = create_image_analysis_figure(encoded_image)
        cell_count = ""
        summary = summary_placeholder
    set_props("detected-cell-count", {"children": cell_count})

    # Reset cell info to placeholders
    set_props("cell-lactate-concentration", {"children": ""})
    set_props("cell-id", {"children": ""})
    set_props("cell-info", {"children": cell_info_processed_placeholder})

    return (
        figure,
        summary,
        hash,
    )


# Process the uncompressed image and saves the processed data
def process_and_save_data(hash: str, image: ImageFile) -> ProcessedData:
    print(f"Processing image with hash {hash}")

    # Process image
    cellpose_model = model_manager.get_cellpose_model()

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

    # Calculate summary statistics
    median_area = np.median([p.area for p in props]) if props else 0

    # Cells by ID
    cells: dict[int, Cell] = {}
    # Cropped images by ID
    cropped_images: dict[int, Image] = {}

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
        if model_manager.has_onnx_model_path():
            model_manager.load_onnx_model_if_needed()
            try:
                # Classify the cell crop
                cell.predicted_properties = model_manager.classify_cell_crop(cropped_image)
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
    aggregate_data = AggregateData(median_area=float(median_area))
    processed_data = ProcessedData(
        cropped_encoded_images=cropped_encoded_images, cells=cells, aggregate_data=aggregate_data
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
def create_summary(processed_data: ProcessedData) -> Any:
    cells = processed_data.cells
    aggregate_data = processed_data.aggregate_data

    summary_start = timer()

    class_counts = {}  # For all classes: {class_name: count}
    concentration_counts = {}  # For concentrations: {concentration: count}
    classification_available = False
    for cell in cells.values():
        predicted_properties = cell.predicted_properties
        if predicted_properties is not None:
            classification_available = True
            predicted_class = predicted_properties.predicted_class
            concentration = predicted_properties.concentration
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            concentration_counts[concentration] = concentration_counts.get(concentration, 0) + 1

    summary_list = [
        html.Li(html.Span(f"Median cell area: {aggregate_data.median_area:.1f} pixels")),
        html.Li(html.Span(f"Mean cell area: {np.mean([cell.area for cell in cells.values()]):.1f} pixels")),
    ]
    if classification_available:
        # Create text summary
        total_cells = sum(class_counts.values())
        summary_list.append(
            html.Li(
                html.Span([
                    "Cell classifications: ",
                    html.Span(f"{total_cells} total cells classified", style={"fontWeight": "bold"}),
                ])
            )
        )
    summary_content: list[ComponentType] = [
        html.Ul(summary_list),
    ]

    if classification_available:
        # Create concentration distribution plot
        if concentration_counts:
            # Ensure all concentration levels are represented
            all_concentrations = [0, 5, 10, 20, 40, 80]
            plot_data = []

            for concentration in all_concentrations:
                count = concentration_counts.get(concentration, 0)
                # Determine class name for this concentration
                class_name = "control_00" if concentration == 0 else f"lactate_{concentration:02d}"

                plot_data.append({
                    "Class": class_name,
                    "Concentration": f"{concentration}",  # Convert to string to use discrete values on x axis
                    "Count": count,
                    "Color": get_concentration_darker_color(concentration),
                })

            df_plot = pd.DataFrame(plot_data)

            # Create bar plot with custom colors
            fig_bar = px.bar(
                df_plot,
                x="Concentration",
                y="Count",
                title="Cell Count by Lactate Concentration",
                labels={"Concentration": "Lactate Concentration [mM]", "Count": "Number of Cells"},
                text="Count",
                text_auto=True,
                color="Color",
                color_discrete_map={row["Color"]: row["Color"] for _, row in df_plot.iterrows()},
            )

            # Update layout
            fig_bar.update_layout(
                height=300,
                margin={"l": 20, "r": 20, "t": 40, "b": 20},
                font={"size": 12},
                showlegend=False,  # Hide color legend since colors are self-explanatory
            )

            # Disable zoom
            fig_bar.update_xaxes(fixedrange=True)
            fig_bar.update_yaxes(fixedrange=True)

            # Add concentration labels on hover
            fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")

            # Add the plot to summary
            summary_content.append(
                dcc.Graph(
                    figure=fig_bar, config={"displayModeBar": False}, style={"height": "320px", "margin": "10px 0"}
                )
            )
    else:
        large_cells = sum(cell.is_large for cell in cells.values())
        small_cells = len(cells) - large_cells
        summary_content.append(
            html.P([
                "Cell count by size: ",
                html.Span(f"{large_cells} large", style={"color": "green", "fontWeight": "bold"}),
                ", ",
                html.Span(f"{small_cells} small", style={"color": "red", "fontWeight": "bold"}),
            ])
        )
    dash.callback_context.record_timing("summary", timer() - summary_start, "Create summary")

    return summary_content


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

    # Determine border color based on prediction or size
    show_cropped_start = timer()
    predicted_properties = cell.predicted_properties
    concentration = None
    if predicted_properties is not None:
        concentration = predicted_properties.concentration
        border_color = get_concentration_darker_color(concentration)
    else:
        border_color = "green" if cell.is_large else "red"

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

    # Add classification results if available
    if predicted_properties is not None:
        cell_info.append(
            html.Div(
                [
                    html.H6("Classification Results:", className="mt-3 mb-2"),
                    html.P(f"Predicted Class: {predicted_properties.predicted_class}", className="fw-bold"),
                    html.P(f"Confidence: {predicted_properties.confidence:.3f}"),
                    html.P(f"Concentration Level: {predicted_properties.concentration}"),
                ],
                className="p-2 border rounded bg-light",
            )
        )
    else:
        cell_info.append(
            html.Div(
                [
                    html.H6("Basic Prediction:", className="mt-3 mb-2"),
                    html.P(f"Area: {int(cell.area)} pixels", className="fw-bold"),
                ],
                className="p-2 border rounded bg-light",
            )
        )

    # Add cell metadata
    cell_info.extend([
        html.H6("Cell Metadata:", className="mt-3 mb-2"),
        html.P(f"Area: {int(cell.area)} pixels"),
        html.P(f"Perimeter: {cell.perimeter:.2f} pixels"),
        html.P(f"Eccentricity: {cell.eccentricity:.3f}"),
        html.P(f"Centroid: ({cell.centroid_x:.1f}, {cell.centroid_y:.1f})"),
    ])

    # Add size classification for non-classified cells
    if predicted_properties is None:
        is_large = cell.is_large
        cell_info.append(
            html.P([
                "Size Classification: ",
                html.Span(
                    "Large" if is_large else "Small",
                    style={"color": "green" if is_large else "red", "fontWeight": "bold"},
                ),
            ])
        )
    dash.callback_context.record_timing("details", timer() - details_start, "Create details")

    set_props(
        "cell-lactate-concentration",
        {"children": f"Lactate concentration: {concentration}mM" if concentration is not None else ""},
    )
    set_props("cell-id", {"children": f"#{cell_id}" if cell_id is not None else ""})

    return cell_info


def main():
    parser = argparse.ArgumentParser(description="Cell Analysis App with Lactate Classification")
    parser.add_argument("--model_path", type=str, default=None, help="Path to ONNX classification model")
    parser.add_argument("--lazy_load", action="store_true", help="Lazily load ONNX classification model")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Set model path if provided
    if args.model_path:
        model_manager.set_onnx_model_path(args.model_path)

    # Load classification model eagerly
    if args.model_path and not args.lazy_load:
        success = model_manager.load_onnx_model_if_needed()
        if success:
            print(f"Successfully loaded classification model from {args.model_path}")
        else:
            print(f"Failed to load classification model from {args.model_path}")
            print("App will run with basic size-based classification only")
    else:
        print("No classification model provided. App will run with basic size-based classification only")

    # Run the app
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
