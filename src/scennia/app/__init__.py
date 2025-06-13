import argparse
from dataclasses import dataclass, field
from timeit import default_timer as timer

import dash
import dash_bootstrap_components as dbc
import diskcache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import DiskcacheManager, dcc, html
from dash.dependencies import Input, Output, State
from PIL.Image import Image
from PIL.ImageFile import ImageFile
from skimage.measure import find_contours, regionprops
from skimage.segmentation import find_boundaries

from scennia.app.image import (
    EncodedImage,
    calculate_image_hash,
    create_complete_figure,
    crop_cell,
    decode_image,
    encode_image,
    get_compressed_cropped_image,
    get_concentration_color,
    get_concentration_color_plotly,
    get_processed_from_cache,
    save_processed_to_cache,
    update_full_figure_layout,
)
from scennia.app.model import model_manager

# Diskcache for non-production apps when developing locally
background_callback_manager = DiskcacheManager(diskcache.Cache("./cache"))

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    background_callback_manager=background_callback_manager,
    suppress_callback_exceptions=True,
)

# Define the client-side callback for toggling annotations
app.clientside_callback(
    """
    function(showAnnotations, figure) {
        if (!figure || !figure.data) {
            return window.dash_clientside.no_update;
        }

        // Create a copy of the figure to avoid modifying the original
        const newFigure = JSON.parse(JSON.stringify(figure));

        // Set visibility for all traces except the first one (which is the image)
        for (let i = 1; i < newFigure.data.length; i++) {
            newFigure.data[i].visible = showAnnotations;
        }

        // Toggle annotation visibility
        if (newFigure.layout && newFigure.layout.annotations) {
            for (let i = 0; i < newFigure.layout.annotations.length; i++) {
                newFigure.layout.annotations[i].visible = showAnnotations;
            }
        }

        return newFigure;
    }
    """,
    Output("cell-visualization-graph", "figure"),
    Input("show-annotations", "value"),
    State("cell-visualization-graph", "figure"),
)


# App layout
def layout():
    title = [
        html.H1("SCENNIA: Prototype Image Analysis Platform", className="my-2 text-center"),
        html.P(
            "A prototype web app of an AI-powered image analysis platform for cultivated "
            "meat cell line development as part of the SCENNIA project, funded by the Bezos Earth Fund.",
            className="text-center",
        ),
    ]
    images_card = dbc.Card(
        style={"height": "100%"},
        children=[
            dbc.CardHeader("Images"),
            dbc.CardBody([]),
        ],
    )
    image_upload_card = dbc.Card(
        style={"height": "100%"},
        children=[
            dbc.CardHeader("Image Upload"),
            dbc.CardBody([
                dcc.Upload(
                    html.Div(["Drag and Drop or ", html.A("Select an Image")]),
                    id="upload-image",
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                    multiple=False,
                    accept="image/*",  # Accept only image files
                ),
                html.Div(id="upload-output"),
                # Process button
                dbc.Button(
                    "Process Image",
                    id="process-button",
                    color="primary",
                    className="mt-3 w-100",
                    disabled=True,
                ),
            ]),
        ],
    )
    image_analysis_card = dbc.Card(
        style={"height": "100%"},
        children=[
            dbc.CardHeader(
                children=dbc.Row(
                    className="justify-content-between",
                    children=[
                        dbc.Col(className="col-auto", children="Image Analysis"),
                        dbc.Col(
                            className="col-auto",
                            children=dbc.Switch(
                                id="show-annotations",
                                label="Show Annotations",
                                value=True,
                                className="mb-0",
                                label_class_name="mb-0",
                            ),
                        ),
                    ],
                ),
            ),
            dbc.CardBody(
                dcc.Loading(  # Loading spinner
                    id="loading-visualization",
                    type="circle",
                    overlay_style={
                        "visibility": "visible",
                        "filter": "blur(3px) opacity(25%)",
                    },
                    children=[
                        dcc.Graph(
                            id="cell-visualization-graph",
                            figure=update_full_figure_layout(go.Figure(), 0, 0),
                            config={"displayModeBar": False, "staticPlot": False},
                            style={"width": "100%", "height": "100%"},
                        ),
                    ],
                ),
            ),
        ],
    )
    summary_card = dbc.Card(
        children=[
            dbc.CardHeader("Summary"),
            dbc.CardBody(
                dcc.Loading(  # Loading spinner
                    id="loading-summary",
                    type="circle",
                    children=[
                        html.Div(
                            html.P("Upload an image and it will be displayed immediately"),
                            id="summary-panel",
                        ),
                    ],
                ),
            ),
        ],
    )
    cell_info_card = dbc.Card(
        style={"height": "100%"},
        children=[
            dbc.CardHeader("Cell Info"),
            dbc.CardBody([
                # Cell image
                dcc.Loading(  # Loading spinner
                    id="loading-cell-image",
                    type="circle",
                    overlay_style={
                        "visibility": "visible",
                        "filter": "blur(3px) opacity(25%)",
                    },
                    children=[
                        html.Div(id="selected-cell-image", className="mb-3"),
                    ],
                ),
                # Cell details
                html.Div(
                    id="selected-cell-details",
                    children=[
                        html.P(
                            "Process an image and then click on a cell to view details",
                            className="text-muted",
                        ),
                    ],
                ),
            ]),
        ],
    )
    model_status = html.Div(id="model-status")
    alerts = html.Div(
        id="status-alert",
        children=[
            dbc.Alert(
                id="image-load-alert",
                color="success",
                dismissable=True,
                is_open=False,
                duration=5000,
            ),
            dbc.Alert(
                id="image-process-alert",
                color="info",
                dismissable=True,
                is_open=False,
                duration=7500,
            ),
        ],
    )
    return dbc.Container(
        fluid=True,
        children=[
            # Row: Title
            dbc.Row(className="row-cols-1 g-2", children=[dbc.Col(title)]),
            # Row: Main content
            dbc.Row(
                className="row-cols-2 g-2 mb-2",
                children=[
                    dbc.Col(width=8, children=[images_card]),
                    dbc.Col(width=4, children=[image_upload_card]),
                    dbc.Col(
                        width=8,
                        children=[
                            dbc.Row(
                                class_name="row-cols-1 g-2",
                                children=[
                                    dbc.Col(width=12, children=[image_analysis_card]),
                                    dbc.Col(width=12, children=[summary_card]),
                                ],
                            )
                        ],
                    ),
                    dbc.Col(width=4, children=[cell_info_card]),
                ],
            ),
            # Row: Footer
            dbc.Row(
                className="row-cols-1 g-2",
                children=[
                    dbc.Col(width=12, children=[model_status]),
                    dbc.Col(width=12, children=alerts),
                ],
            ),
            # Hidden data stores
            dcc.Store(id="image-hash-store"),
        ],
    )


app.layout = layout()


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


# Server-side stores
@dataclass
class ImageData:
    uncompressed_image: ImageFile
    encoded_image: EncodedImage
    cropped_encoded_images: dict[str, EncodedImage] = field(default_factory=dict)
    cell_data: list = field(default_factory=list)
    mask_data: dict = field(default_factory=dict)


IMAGE_DATA_STORE: dict[str, ImageData] = {}


# Display uploaded image
@app.callback(
    [
        Output("cell-visualization-graph", "figure", allow_duplicate=True),
        Output("upload-output", "children"),
        Output("image-hash-store", "data"),
        Output("image-load-alert", "children"),
        Output("image-load-alert", "color"),
        Output("image-load-alert", "is_open"),
    ],
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    background=False,
    running=[  # Disable upload and process while processing.
        (Output("upload-image", "disabled"), True, False),
        (Output("process-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def display_uploaded_image(contents, filename):
    print(f"Upload callback triggered. Contents: {'Present' if contents else 'None'}, Filename: {filename}")

    if contents is None:
        return html.Div("Upload an image"), html.Div("No image uploaded yet"), None, None, None

    try:
        print(f"Processing uploaded file: {filename}")

        # Decode uploaded image
        decode_start = timer()
        image = decode_image(contents)
        dash.callback_context.record_timing("decode", timer() - decode_start, "Decode uploaded image")

        # Hash uploaded image
        hash_start = timer()
        image_hash = calculate_image_hash(image)
        dash.callback_context.record_timing("hash", timer() - hash_start, "Hash uploaded image")

        # Encode to WebP
        encode_start = timer()
        encoded_image = encode_image(image)
        dash.callback_context.record_timing("encode", timer() - encode_start, "Encode uploaded image")

        # Create a figure with just the original image
        figure_start = timer()
        fig = go.Figure()
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
        update_full_figure_layout(fig, image.width, image.height, False, False)
        dash.callback_context.record_timing("figure", timer() - figure_start, "Create figure")

        # Update store
        IMAGE_DATA_STORE[image_hash] = ImageData(image, encoded_image)

        return (
            fig,
            html.Div(f"Uploaded: {filename}"),
            image_hash,
            # Status alert
            f"Image loaded successfully: {filename}",
            "success",
            True,
        )

    except Exception as e:
        print(f"Error displaying uploaded image: {e!s}")

        return (
            html.Div("Error displaying image"),
            html.Div("Error processing upload"),
            None,
            None,
            # Status alert
            f"Error displaying image: {e!s}",
            "danger",
            True,
        )


# Process image and check cache
@app.callback(
    [
        Output("summary-panel", "children"),
        Output("cell-visualization-graph", "figure", allow_duplicate=True),
        Output("image-process-alert", "children"),
        Output("image-process-alert", "color"),
        Output("image-process-alert", "is_open"),
    ],
    Input("process-button", "n_clicks"),
    [
        State("image-hash-store", "data"),
        State("show-annotations", "value"),
    ],
    background=False,
    running=[  # Disable upload and process while processing.
        (Output("upload-image", "disabled"), True, False),
        (Output("process-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
)
def process_image(n_clicks, image_hash, show_annotations):
    if n_clicks is None or image_hash is None or image_hash not in IMAGE_DATA_STORE:
        return (None, None, html.P("Upload an image and click 'Process Image'"), dash.no_update, None, dash.no_update)

    image_data = IMAGE_DATA_STORE[image_hash]

    load_start = timer()
    cached_data = get_processed_from_cache(image_hash)
    dash.callback_context.record_timing("load_cache_data", timer() - load_start, "Read and deserialize cached data")

    # Check cache first
    if cached_data:
        print("Using cached results")

        processing_start = timer()

        # Create summary from cached data
        cell_props = cached_data["cell_data"]
        mask_data = cached_data["mask_data"]
        median_area = mask_data["median_area"]

        # Count cells by classification if available
        class_counts = {}  # For all classes: {class_name: count}
        concentration_counts = {}  # For concentrations: {concentration: count}
        classification_available = False

        for cell in cell_props:
            predicted_props = cell.get("predicted_properties", {})
            if "predicted_class" in predicted_props:
                classification_available = True
                predicted_class = predicted_props["predicted_class"]
                concentration = predicted_props.get("concentration", 0)

                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                concentration_counts[concentration] = concentration_counts.get(concentration, 0) + 1

        dash.callback_context.record_timing("total_processing", timer() - processing_start, "Total processing (cached)")

        # Create summary content
        summary_start = timer()
        summary_content = [
            html.H5(f"Detected {len(cell_props)} cells (from cache)"),
            html.P(f"Median cell area: {median_area:.1f} pixels"),
            html.P(f"Mean cell area: {np.mean([c['area'] for c in cell_props]):.1f} pixels"),
        ]

        if classification_available:
            # Create text summary
            total_cells = sum(class_counts.values())
            summary_content.append(
                html.P([
                    "Cell classifications: ",
                    html.Span(f"{total_cells} total cells classified", style={"fontWeight": "bold"}),
                ])
            )

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
                        "Concentration": concentration,
                        "Count": count,
                        "Color": get_concentration_color_plotly(concentration),
                    })

                df_plot = pd.DataFrame(plot_data)

                # Create bar plot with custom colors
                fig_bar = px.bar(
                    df_plot,
                    x="Concentration",
                    y="Count",
                    title="Cell Count by Concentration Level",
                    labels={"Concentration": "Concentration Level", "Count": "Number of Cells"},
                    color="Color",
                    color_discrete_map={row["Color"]: row["Color"] for _, row in df_plot.iterrows()},
                )

                # Update layout
                fig_bar.update_layout(
                    height=300,
                    margin={"l": 20, "r": 20, "t": 40, "b": 20},
                    font={"size": 12},
                    showlegend=False,  # Hide color legend since colors are self-explanatory
                    xaxis={
                        "tickmode": "array",
                        "tickvals": all_concentrations,
                        "ticktext": [f"{c}" for c in all_concentrations],
                        "title": "Concentration Level",
                    },
                )

                # Add concentration labels on hover
                fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")

                # Add the plot to summary
                summary_content.append(
                    dcc.Graph(
                        figure=fig_bar, config={"displayModeBar": False}, style={"height": "320px", "margin": "10px 0"}
                    )
                )
        else:
            large_cells = sum(c["is_large"] for c in cell_props)
            small_cells = len(cell_props) - large_cells
            summary_content.append(
                html.P([
                    "Cell count by size: ",
                    html.Span(f"{large_cells} large", style={"color": "green", "fontWeight": "bold"}),
                    ", ",
                    html.Span(f"{small_cells} small", style={"color": "red", "fontWeight": "bold"}),
                ])
            )

        summary_content.extend([
            html.P(html.B("Click on any cell marker to view details"), className="mt-3"),
        ])
        dash.callback_context.record_timing("summary", timer() - summary_start, "Create summary")

        # Create the figure with cell data
        figure_start = timer()
        fig = create_complete_figure(image_data.encoded_image, cell_props, show_annotations)
        dash.callback_context.record_timing("figure", timer() - figure_start, "Create figure")

        # Update stores
        image_data.cell_data = cell_props
        image_data.mask_data = mask_data

        return (
            summary_content,
            fig,
            # Status alert
            f"Processed image from cache: {len(cell_props)} cells detected",
            "info",
            True,
        )

    try:
        cellpose_model = model_manager.get_cellpose_model()

        processing_start = timer()

        # Convert uncompressed image to array for processing
        uncompressed_image = image_data.uncompressed_image
        image_array = np.asarray(uncompressed_image)

        # Process with cellpose (using default values)
        flow_threshold = 0.4  # Default value
        cell_prob_threshold = 0.0  # Default value

        segmentation_start = timer()
        result = cellpose_model.eval(
            [image_array], flow_threshold=flow_threshold, cellprob_threshold=cell_prob_threshold
        )
        dash.callback_context.record_timing("segmentation", timer() - segmentation_start, "Segmentation")

        mask = result[0][0]
        props = regionprops(mask)

        # Calculate summary statistics
        median_area = np.median([p.area for p in props]) if props else 0

        # Store cell properties in a suitable format
        cell_data = []
        predicted_properties = {}  # This will store all predicted properties
        cropped_uncompressed_images: dict[str, Image] = {}  # This will store cropped images

        crop_and_classify_start = timer()
        for i, prop in enumerate(props):
            cell_id = i + 1
            is_large = prop.area > median_area

            # Create basic cell data first
            cell_props = {
                "id": cell_id,
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "area": float(prop.area),
                "perimeter": float(prop.perimeter),
                "eccentricity": float(prop.eccentricity),
                "bbox": [int(x) for x in prop.bbox],
                "label": int(prop.label),
                "is_large": bool(is_large),
            }

            # Contour
            cell_props["contour"] = find_contours(mask == cell_id)[0].T.tolist()

            # Crop cell
            cropped_uncompressed_image = crop_cell(uncompressed_image, cell_props["bbox"])
            cropped_uncompressed_images[str(cell_id)] = cropped_uncompressed_image

            # Perform classification if model is available
            cell_prediction = {}
            if model_manager.has_onnx_model_path():
                model_manager.load_onnx_model_if_needed()
                try:
                    # Classify the cell crop
                    classification_result = model_manager.classify_cell_crop(cropped_uncompressed_image)

                    if "error" not in classification_result:
                        cell_prediction = classification_result
                    else:
                        print(f"Classification error for cell {cell_id}: {classification_result['error']}")
                        # Fallback to basic size prediction
                        cell_prediction = {"size_pixels": float(prop.area)}

                except Exception as e:
                    print(f"Error classifying cell {cell_id}: {e}")
                    # Fallback to basic size prediction
                    cell_prediction = {"size_pixels": float(prop.area)}
            else:
                # No classification model, use basic size prediction
                cell_prediction = {"size_pixels": float(prop.area)}

            # Store the predicted properties
            predicted_properties[str(cell_id)] = cell_prediction
            cell_props["predicted_properties"] = cell_prediction

            cell_data.append(cell_props)

        dash.callback_context.record_timing("crop_and_classify", timer() - crop_and_classify_start, "Crop and classify")
        dash.callback_context.record_timing("total_processing", timer() - processing_start, "Total processing")

        # Store mask data
        save_start = timer()
        mask_data = {
            "mask": mask.tolist(),
            "median_area": float(median_area),
            "boundary": find_boundaries(mask, mode="inner").tolist(),
        }

        # Save to cache with predicted properties and cropped images
        save_processed_to_cache(image_hash, cell_data, mask_data, predicted_properties, cropped_uncompressed_images)
        dash.callback_context.record_timing("save_cache_data", timer() - save_start, "Serialize and write cache data")

        # Create summary content
        summary_start = timer()
        class_counts = {}  # For all classes: {class_name: count}
        concentration_counts = {}  # For concentrations: {concentration: count}
        classification_available = False

        for cell in cell_data:
            predicted_props = cell.get("predicted_properties", {})
            if "predicted_class" in predicted_props:
                classification_available = True
                predicted_class = predicted_props["predicted_class"]
                concentration = predicted_props.get("concentration", 0)

                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                concentration_counts[concentration] = concentration_counts.get(concentration, 0) + 1

        summary_content = [
            html.H5(f"Detected {len(cell_data)} cells"),
            html.P(f"Median cell area: {median_area:.1f} pixels"),
            html.P(f"Mean cell area: {np.mean([c['area'] for c in cell_data]):.1f} pixels"),
        ]

        if classification_available:
            # Create text summary
            total_cells = sum(class_counts.values())
            summary_content.append(
                html.P([
                    "Cell classifications: ",
                    html.Span(f"{total_cells} total cells classified", style={"fontWeight": "bold"}),
                ])
            )

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
                        "Concentration": concentration,
                        "Count": count,
                        "Color": get_concentration_color_plotly(concentration),
                    })

                df_plot = pd.DataFrame(plot_data)

                # Create bar plot with custom colors
                fig_bar = px.bar(
                    df_plot,
                    x="Concentration",
                    y="Count",
                    title="Cell Count by Concentration Level",
                    labels={"Concentration": "Concentration Level", "Count": "Number of Cells"},
                    color="Color",
                    color_discrete_map={row["Color"]: row["Color"] for _, row in df_plot.iterrows()},
                )

                # Update layout
                fig_bar.update_layout(
                    height=300,
                    margin={"l": 20, "r": 20, "t": 40, "b": 20},
                    font={"size": 12},
                    showlegend=False,  # Hide color legend since colors are self-explanatory
                    xaxis={
                        "tickmode": "array",
                        "tickvals": all_concentrations,
                        "ticktext": [f"{c}" for c in all_concentrations],
                        "title": "Concentration Level",
                    },
                )

                # Add concentration labels on hover
                fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")

                # Add the plot to summary
                summary_content.append(
                    dcc.Graph(
                        figure=fig_bar, config={"displayModeBar": False}, style={"height": "320px", "margin": "10px 0"}
                    )
                )
        else:
            large_cells = sum(c["is_large"] for c in cell_data)
            small_cells = len(cell_data) - large_cells
            summary_content.append(
                html.P([
                    "Cell count by size: ",
                    html.Span(f"{large_cells} large", style={"color": "green", "fontWeight": "bold"}),
                    ", ",
                    html.Span(f"{small_cells} small", style={"color": "red", "fontWeight": "bold"}),
                ])
            )

        summary_content.append(html.P(html.B("Click on any cell marker to view details"), className="mt-3"))
        dash.callback_context.record_timing("summary", timer() - summary_start, "Create summary")

        # Create the figure with cell data
        figure_start = timer()
        fig = create_complete_figure(image_data.encoded_image, cell_data, show_annotations)
        dash.callback_context.record_timing("figure", timer() - figure_start, "Create figure")

        # Update stores
        image_data.cell_data = cell_data
        image_data.mask_data = mask_data

        return (
            summary_content,
            fig,
            # Status alert
            f"Processed image: {len(cell_data)} cells detected",
            "success",
            True,
        )

    except Exception as e:
        print(f"Error processing image: {e!s}")

        return (
            None,
            None,
            html.P(f"Error processing image: {e!s}"),
            dash.no_update,
            None,
            # Status alert
            f"Error processing image: {e!s}",
            "danger",
            True,
        )


# Callback to show cell details when clicked
@app.callback(
    [Output("selected-cell-image", "children"), Output("selected-cell-details", "children")],
    Input("cell-visualization-graph", "clickData"),
    [
        State("image-hash-store", "data"),
    ],
    prevent_initial_call=True,
)
def display_selected_cell(click_data, image_hash):
    if not click_data or image_hash not in IMAGE_DATA_STORE:
        return html.Div(), html.P("Process an image and then click on a cell to view details", className="text-muted")

    image_data = IMAGE_DATA_STORE[image_hash]
    cropped_encoded_images = image_data.cropped_encoded_images
    cell_data = image_data.cell_data

    cell_start = timer()
    try:
        # Get click coordinates
        if "points" not in click_data or not click_data["points"] or len(click_data["points"]) == 0:
            return html.Div(), html.P("Click on a cell to view its details", className="text-muted")

        point = click_data["points"][0]

        # Check for customdata
        if "customdata" not in point:
            # No customdata, use closest cell approach
            if "x" not in point or "y" not in point:
                return html.Div(), html.P("Click on a cell to view its details", className="text-muted")

            click_x = point["x"]
            click_y = point["y"]

            # Find the closest cell to the click coordinates
            closest_cell = None
            min_distance = float("inf")

            for cell in cell_data:
                dx = cell["centroid_x"] - click_x
                dy = cell["centroid_y"] - click_y
                distance = dx * dx + dy * dy  # Squared distance is enough for comparison

                if distance < min_distance:
                    min_distance = distance
                    closest_cell = cell

            # Set a maximum distance threshold (radius squared)
            max_distance_threshold = 30 * 30  # 30 pixel radius
            if min_distance > max_distance_threshold:
                return html.Div(), html.P("Click closer to a cell center to view its details", className="text-muted")

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
                    return html.Div(), html.P("Invalid click data", className="text-muted")
            else:
                # Direct value
                cell_id = customdata

            # Find the cell in our data
            cell = next((c for c in cell_data if c["id"] == cell_id), None)

        dash.callback_context.record_timing("cell", timer() - cell_start, "Find cell")

        # If no cell was found, show a placeholder.
        if not cell:
            return html.Div(), html.P(f"Cell data not found for ID {cell_id}", className="text-muted")

        # Get cropped image
        cropped_start = timer()
        cell_id = str(cell["id"])
        if cell_id in cropped_encoded_images:
            cropped_encoded_image = cropped_encoded_images[cell_id]
        else:
            cropped_image = get_compressed_cropped_image(image_hash, cell_id)
            cropped_encoded_image = encode_image(cropped_image) if cropped_image else None
            if cropped_encoded_image:
                # Update store
                cropped_encoded_images[cell_id] = cropped_encoded_image
        dash.callback_context.record_timing("get_cropped", timer() - cropped_start, "Get cropped image")

        # Determine border color based on prediction or size
        show_cropped_start = timer()
        predicted_props = cell.get("predicted_properties", {})
        if "concentration" in predicted_props:
            concentration = predicted_props.get("concentration", 0)
            border_color = get_concentration_color(concentration)
        else:
            border_color = "green" if cell["is_large"] else "red"

        if cropped_encoded_image:
            # Use the cached cropped image directly
            cell_image = html.Img(
                src=cropped_encoded_image.contents,
                style={"width": "100%", "border": f"3px solid {border_color}"},
            )
        else:
            print("Creating cropped image dynamically (fallback)")

            uncompressed_image = image_data.uncompressed_image

            # Create the cropped image dynamically (fallback)
            y0, x0, y1, x1 = cell["bbox"]
            padding = 10
            y0 = max(0, y0 - padding)
            x0 = max(0, x0 - padding)
            y1 = min(uncompressed_image.height, y1 + padding)
            x1 = min(uncompressed_image.width, x1 + padding)

            # Create a zoomed-in view of the cell
            cell_fig = go.Figure()

            # Add the original image
            cell_fig.add_layout_image({
                "source": uncompressed_image,  # TODO: use encoded image
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
        cell_details: list = [
            html.H5(f"Cell {cell['id']} Details", style={"color": border_color}),
        ]

        # Add classification results if available
        if "predicted_class" in predicted_props:
            cell_details.append(
                html.Div(
                    [
                        html.H6("Classification Results:", className="mt-3 mb-2"),
                        html.P(f"Predicted Class: {predicted_props['predicted_class']}", className="fw-bold"),
                        html.P(f"Confidence: {predicted_props.get('confidence', 0):.3f}"),
                        html.P(f"Concentration Level: {predicted_props.get('concentration', 0)}"),
                    ],
                    className="p-2 border rounded bg-light",
                )
            )
        else:
            cell_details.append(
                html.Div(
                    [
                        html.H6("Basic Prediction:", className="mt-3 mb-2"),
                        html.P(
                            f"Size: {int(predicted_props.get('size_pixels', cell['area']))} pixels", className="fw-bold"
                        ),
                    ],
                    className="p-2 border rounded bg-light",
                )
            )

        # Add cell metadata
        cell_details.extend([
            html.H6("Cell Metadata:", className="mt-3 mb-2"),
            html.P(f"Area: {int(cell['area'])} pixels"),
            html.P(f"Perimeter: {cell['perimeter']:.2f} pixels"),
            html.P(f"Eccentricity: {cell['eccentricity']:.3f}"),
            html.P(f"Centroid: ({cell['centroid_x']:.1f}, {cell['centroid_y']:.1f})"),
        ])

        # Add size classification for non-classified cells
        if "predicted_class" not in predicted_props:
            is_large = cell["is_large"]
            cell_details.append(
                html.P([
                    "Size Classification: ",
                    html.Span(
                        "Large" if is_large else "Small",
                        style={"color": "green" if is_large else "red", "fontWeight": "bold"},
                    ),
                ])
            )

        dash.callback_context.record_timing("details", timer() - details_start, "Create details")

        return cell_image, cell_details

    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        print(f"Error in display_selected_cell: {e!s}")
        print(traceback_str)
        return html.Div(), html.P(f"Error: {e!s}", className="text-danger")


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
