import argparse
import base64
import hashlib
import io
import json
import os
import time
from io import BytesIO

import dash
import dash_bootstrap_components as dbc
import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from cellpose import models
from dash import dcc, html
from dash.dependencies import ClientsideFunction, Input, Output, State
from PIL import Image
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from torchvision import transforms

# Simple file-based cache system
CACHE_DIR = "cache"
CROPPED_CACHE_DIR = os.path.join(CACHE_DIR, "cropped")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CROPPED_CACHE_DIR, exist_ok=True)


class ModelManager:
    """Manages ONNX model loading and classification without global variables"""

    def __init__(self):
        self.classification_model = None
        self.model_metadata = None
        self.transform_for_classification = None

    def load_model(self, model_path):
        """Load ONNX classification model and metadata"""
        try:
            # Load ONNX model
            self.classification_model = ort.InferenceSession(model_path)
            print(f"Loaded classification model from {model_path}")

            # Load metadata
            metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)
                print(f"Loaded model metadata: {self.model_metadata['num_classes']} classes")
                print(f"Classes: {self.model_metadata['class_names']}")
            else:
                print("Warning: No metadata file found")
                self.model_metadata = None

            # Setup transforms for classification
            img_size = self.model_metadata.get("img_size", 224) if self.model_metadata else 224
            mean = (
                self.model_metadata.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
                if self.model_metadata
                else [0.485, 0.456, 0.406]
            )
            std = (
                self.model_metadata.get("normalization", {}).get("std", [0.229, 0.224, 0.225])
                if self.model_metadata
                else [0.229, 0.224, 0.225]
            )

            self.transform_for_classification = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            return True

        except Exception as e:
            print(f"Error loading classification model: {e}")
            return False

    def classify_cell_crop(self, cell_crop_pil):
        """Classify a cell crop using the loaded ONNX model"""
        if self.classification_model is None or self.transform_for_classification is None:
            return {"error": "Classification model not loaded"}

        try:
            # Ensure image is RGB
            if cell_crop_pil.mode != "RGB":
                cell_crop_pil = cell_crop_pil.convert("RGB")

            # Apply transforms
            input_tensor = self.transform_for_classification(cell_crop_pil).unsqueeze(0).numpy()

            # Run inference
            inputs = {self.classification_model.get_inputs()[0].name: input_tensor}
            outputs = self.classification_model.run(None, inputs)
            predictions = outputs[0][0]  # Get first batch item

            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions))

            # Get class name if metadata available - Combined if statement (fixes SIM102)
            class_name = "Unknown"
            if (
                self.model_metadata
                and "class_names" in self.model_metadata
                and predicted_class_idx < len(self.model_metadata["class_names"])
            ):
                class_name = self.model_metadata["class_names"][predicted_class_idx]

            # Parse treatment and concentration from class name
            treatment_type = "unknown"
            concentration = 0

            if "_" in class_name:
                parts = class_name.split("_")
                if len(parts) >= 2:
                    treatment_type = parts[0]
                    try:
                        concentration = int(parts[1])
                    except ValueError:
                        concentration = 0

            return {
                "predicted_class": class_name,
                "predicted_class_idx": int(predicted_class_idx),
                "confidence": confidence,
                "treatment_type": treatment_type,
                "concentration": concentration,
                "all_predictions": predictions.tolist(),
            }

        except Exception as e:
            print(f"Error in cell classification: {e}")
            return {"error": str(e)}

    def is_loaded(self):
        """Check if model is loaded and ready"""
        return self.classification_model is not None and self.transform_for_classification is not None


# Create global instance to replace global variables
model_manager = ModelManager()


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


def load_classification_model(model_path):
    """Load ONNX classification model and metadata"""
    return model_manager.load_model(model_path)


def classify_cell_crop(cell_crop_pil):
    """Classify a cell crop using the loaded ONNX model"""
    return model_manager.classify_cell_crop(cell_crop_pil)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the client-side callback for toggling annotations
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="toggleAnnotations"),
    Output("cell-visualization-graph", "figure"),
    Input("show-annotations", "value"),
    State("cell-visualization-graph", "figure"),
)

# Add the JavaScript code for the client-side function
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
            window.dash_clientside = Object.assign({}, window.dash_clientside, {
                clientside: {
                    toggleAnnotations: function(showAnnotations, figure) {
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
                }
            });
        </script>
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

# Keep GPU enabled as requested
model = models.CellposeModel(gpu=True)

# App layout
app.layout = html.Div(
    [
        dbc.Container(
            [
                html.H1("Interactive Cell Analyzer with Lactase Classification", className="my-4 text-center"),
                # Add model status indicator
                html.Div(id="model-status", className="mb-3"),
                dbc.Row(
                    [
                        # Left column - Controls and Cell Details
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Upload"),
                                        dbc.CardBody(
                                            [
                                                dcc.Upload(
                                                    id="upload-image",
                                                    children=html.Div(["Drag and Drop or ", html.A("Select an Image")]),
                                                    style={
                                                        "width": "100%",
                                                        "height": "60px",
                                                        "lineHeight": "60px",
                                                        "borderWidth": "1px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "5px",
                                                        "textAlign": "center",
                                                        "margin": "10px 0",
                                                    },
                                                    multiple=False,
                                                    accept="image/*",  # Accept only image files
                                                ),
                                                html.Div(id="upload-output"),
                                                # Process button - will use default values for segmentation
                                                dbc.Button(
                                                    "Process Image",
                                                    id="process-button",
                                                    color="primary",
                                                    className="mt-3 w-100",
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                # Selected cell details panel
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Selected Cell"),
                                        dbc.CardBody(
                                            [
                                                # Add loading spinner for cell image
                                                dcc.Loading(
                                                    id="loading-cell-image",
                                                    type="circle",
                                                    children=html.Div(id="selected-cell-image", className="mb-3"),
                                                ),
                                                # Add loading spinner for cell details
                                                html.Div(
                                                    id="selected-cell-details",
                                                    children=[
                                                        html.P(
                                                            "Process an image and then click on a cell to view details",
                                                            className="text-muted",
                                                        )
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mt-3",
                                ),
                            ],
                            width=4,
                        ),
                        # Right column - Visualization
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col("Cell Visualization", width=8),
                                                        dbc.Col(
                                                            [
                                                                dbc.Form(
                                                                    [
                                                                        dbc.Switch(
                                                                            id="show-annotations",
                                                                            label="Show Annotations",
                                                                            value=True,
                                                                            className="ms-2",
                                                                        ),
                                                                    ],
                                                                    className="d-flex align-items-center",
                                                                )
                                                            ],
                                                            width=4,
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                # This div will contain our visualization
                                                html.Div(
                                                    id="visualization-container",
                                                    children=[
                                                        dcc.Loading(
                                                            id="loading-visualization",
                                                            type="circle",
                                                            children=html.Div(id="visualization-output"),
                                                        )
                                                    ],
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                                # Summary panel
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Summary"),
                                        dbc.CardBody(
                                            dcc.Loading(
                                                id="loading-summary",
                                                type="circle",
                                                children=html.Div(
                                                    id="summary-panel",
                                                    children=[
                                                        html.P("Upload an image and it will be displayed immediately")
                                                    ],
                                                ),
                                            )
                                        ),
                                    ],
                                    className="mt-3",
                                ),
                            ],
                            width=8,
                        ),
                    ]
                ),
                # Add status alert at the bottom
                html.Div(id="status-alert", className="mt-3"),
            ],
            fluid=True,
        ),
        # Hidden data stores
        dcc.Store(id="cell-data-store"),
        dcc.Store(id="processed-image-store"),
        dcc.Store(id="mask-data-store"),
        dcc.Store(id="image-hash-store"),
        dcc.Store(id="cropped-images-store"),  # Store for cropped images
    ]
)


# Add callback for model status
@app.callback(
    Output("model-status", "children"),
    Input("upload-image", "id"),  # Trigger on app load by using a static component ID
)
def display_model_status(_):
    if model_manager.is_loaded() and model_manager.model_metadata is not None:
        return dbc.Alert(
            [
                html.Strong("Classification Model Loaded: "),
                f"{model_manager.model_metadata.get('model_name', 'Unknown')} with {model_manager.model_metadata.get('num_classes', 0)} classes",
                html.Br(),
                html.Small(f"Classes: {', '.join(model_manager.model_metadata.get('class_names', []))}"),
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
                    "visible": bool(cell_data and show_annotations),
                }
            ]
            if cell_data
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


# Display uploaded image
@app.callback(
    [
        Output("visualization-output", "children"),
        Output("upload-output", "children"),
        Output("processed-image-store", "data"),
        Output("image-hash-store", "data"),
        Output("status-alert", "children"),
    ],
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    prevent_initial_call=True,
)
def display_uploaded_image(contents, filename):
    print(f"Upload callback triggered. Contents: {'Present' if contents else 'None'}, Filename: {filename}")

    if contents is None:
        return html.Div("Upload an image"), html.Div("No image uploaded yet"), None, None, None

    try:
        print(f"Processing uploaded file: {filename}")

        # Parse the image
        img = parse_image(contents)

        # Calculate image hash for caching
        img_hash = calculate_image_hash(img)

        # Convert to base64 for display
        encoded_image = numpy_to_b64(img)

        # Get image dimensions
        height, width = img.shape[:2]

        # Create a figure with just the original image
        fig = go.Figure()

        # Add the original image
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

        # Update layout
        fig.update_layout(
            autosize=True,
            height=600,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
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

        # Create the graph object
        visualization = dcc.Graph(
            id="cell-visualization-graph",
            figure=fig,
            config={"displayModeBar": False, "staticPlot": False},
            style={"width": "100%", "height": "100%"},
        )

        # Display a status alert
        status = dbc.Alert(f"Image loaded successfully: {filename}", color="success", dismissable=True, is_open=True)

        return visualization, html.Div(f"Uploaded: {filename}"), encoded_image, img_hash, status

    except Exception as e:
        print(f"Error displaying uploaded image: {e!s}")

        # Display error alert
        error_alert = dbc.Alert(f"Error displaying image: {e!s}", color="danger", dismissable=True, is_open=True)

        return html.Div("Error displaying image"), html.Div("Error processing upload"), None, None, error_alert


# Process image and check cache
@app.callback(
    [
        Output("cell-data-store", "data"),
        Output("mask-data-store", "data"),
        Output("summary-panel", "children"),
        Output("visualization-output", "children", allow_duplicate=True),
        Output("cropped-images-store", "data"),
        Output("status-alert", "children", allow_duplicate=True),
    ],
    Input("process-button", "n_clicks"),
    [State("processed-image-store", "data"), State("image-hash-store", "data"), State("show-annotations", "value")],
    prevent_initial_call=True,
)
def process_image(n_clicks, encoded_image, img_hash, show_annotations):
    if n_clicks is None or encoded_image is None or img_hash is None:
        return (None, None, html.P("Upload an image and click 'Process Image'"), dash.no_update, None, dash.no_update)

    start_time = time.time()

    # Check cache first
    cached_data = get_from_cache(img_hash)
    if cached_data:
        print("Using cached results")

        # Create summary from cached data
        cell_data = cached_data["cell_data"]
        mask_data = cached_data["mask_data"]

        # Load cropped images from disk cache
        cropped_images = {}
        for cell in cell_data:
            cell_id = cell["id"]
            cached_crop = get_cropped_image(img_hash, cell_id)
            if cached_crop:
                cropped_images[cell_id] = cached_crop

        median_area = mask_data["median_area"]

        # Count cells by classification if available
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

        processing_time = time.time() - start_time

        # Create summary content
        summary_content = [
            html.H5(f"Detected {len(cell_data)} cells (from cache)"),
            html.P(f"Median cell area: {median_area:.1f} pixels"),
            html.P(f"Mean cell area: {np.mean([c['area'] for c in cell_data]):.1f} pixels"),
        ]

        if classification_available:
            # Create text summary
            total_cells = sum(class_counts.values())
            summary_content.append(
                html.P(
                    [
                        "Cell classifications: ",
                        html.Span(f"{total_cells} total cells classified", style={"fontWeight": "bold"}),
                    ]
                )
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

                    plot_data.append(
                        {
                            "Class": class_name,
                            "Concentration": concentration,
                            "Count": count,
                            "Color": get_concentration_color_plotly(concentration),
                        }
                    )

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
                html.P(
                    [
                        "Cell count by size: ",
                        html.Span(f"{large_cells} large", style={"color": "green", "fontWeight": "bold"}),
                        ", ",
                        html.Span(f"{small_cells} small", style={"color": "red", "fontWeight": "bold"}),
                    ]
                )
            )

        summary_content.extend(
            [
                html.P(f"Processing time: {processing_time:.2f} seconds (cached)", className="text-muted mt-2"),
                html.P(html.B("Click on any cell marker to view details"), className="mt-3"),
            ]
        )

        # Create the figure with cell data
        fig = create_complete_figure(encoded_image, cell_data, mask_data, show_annotations)

        # Create the graph object
        visualization = dcc.Graph(
            id="cell-visualization-graph",
            figure=fig,
            config={"displayModeBar": False, "staticPlot": False},
            style={"width": "100%", "height": "100%"},
        )

        # Success status
        status = dbc.Alert(
            f"Processed image from cache: {len(cell_data)} cells detected in {processing_time:.2f} seconds",
            color="info",
            dismissable=True,
            is_open=True,
        )

        return cell_data, mask_data, summary_content, visualization, cropped_images, status

    try:
        processing_start = time.time()

        # Parse the image for processing
        content_type, content_string = encoded_image.split(",")
        decoded = base64.b64decode(content_string)
        img = np.array(Image.open(io.BytesIO(decoded)))

        # Process with cellpose (using default values)
        flow_threshold = 0.4  # Default value
        cell_prob_threshold = 0.0  # Default value

        segmentation_start = time.time()
        result = model.eval([img], flow_threshold=flow_threshold, cellprob_threshold=cell_prob_threshold)
        segmentation_time = time.time() - segmentation_start

        mask = result[0][0]
        props = regionprops(mask)

        # Calculate summary statistics
        median_area = np.median([p.area for p in props]) if props else 0

        # Store cell properties in a suitable format
        cell_props = []
        predicted_properties = {}  # This will store all predicted properties
        cropped_images = {}  # This will store cropped images

        cropping_start = time.time()
        classification_start = time.time()

        for i, prop in enumerate(props):
            cell_id = i + 1
            is_large = prop.area > median_area

            # Create basic cell data first
            cell_data = {
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

            # Create cropped image
            cropped_img_b64 = create_cell_crop(encoded_image, cell_data, {"mask": mask})
            cropped_images[cell_id] = cropped_img_b64

            # Perform classification if model is available
            cell_prediction = {}
            if model_manager.is_loaded():
                try:
                    # Convert base64 to PIL Image for classification
                    crop_content_type, crop_content_string = cropped_img_b64.split(",")
                    crop_decoded = base64.b64decode(crop_content_string)
                    crop_pil = Image.open(io.BytesIO(crop_decoded))

                    # Classify the cell crop
                    classification_result = classify_cell_crop(crop_pil)

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
            cell_data["predicted_properties"] = cell_prediction

            cell_props.append(cell_data)

        classification_time = time.time() - classification_start
        cropping_time = time.time() - cropping_start

        # Store mask data
        mask_data = {
            "mask": mask.tolist(),
            "median_area": float(median_area),
            "boundary": find_boundaries(mask, mode="inner").tolist(),
        }

        # Save to cache with predicted properties and cropped images
        cache_start = time.time()
        save_to_cache(img_hash, cell_props, encoded_image, mask_data, predicted_properties, cropped_images)
        cache_time = time.time() - cache_start

        # Create summary content
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

        total_processing_time = time.time() - processing_start

        summary_content = [
            html.H5(f"Detected {len(cell_props)} cells"),
            html.P(f"Median cell area: {median_area:.1f} pixels"),
            html.P(f"Mean cell area: {np.mean([c['area'] for c in cell_props]):.1f} pixels"),
        ]

        if classification_available:
            # Create text summary
            total_cells = sum(class_counts.values())
            summary_content.append(
                html.P(
                    [
                        "Cell classifications: ",
                        html.Span(f"{total_cells} total cells classified", style={"fontWeight": "bold"}),
                    ]
                )
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

                    plot_data.append(
                        {
                            "Class": class_name,
                            "Concentration": concentration,
                            "Count": count,
                            "Color": get_concentration_color_plotly(concentration),
                        }
                    )

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
                html.P(
                    [
                        "Cell count by size: ",
                        html.Span(f"{large_cells} large", style={"color": "green", "fontWeight": "bold"}),
                        ", ",
                        html.Span(f"{small_cells} small", style={"color": "red", "fontWeight": "bold"}),
                    ]
                )
            )

        # Add timing information
        summary_content.append(
            html.Div(
                [
                    html.P("Processing times:", className="fw-bold mt-2 mb-1"),
                    html.Ul(
                        [
                            html.Li(f"Segmentation: {segmentation_time:.2f}s"),
                            html.Li(f"Classification: {classification_time:.2f}s"),
                            html.Li(f"Cropping: {cropping_time:.2f}s"),
                            html.Li(f"Caching: {cache_time:.2f}s"),
                            html.Li(f"Total: {total_processing_time:.2f}s"),
                        ],
                        className="small text-muted",
                    ),
                ]
            )
        )

        summary_content.append(html.P(html.B("Click on any cell marker to view details"), className="mt-3"))

        # Create the figure with cell data
        fig = create_complete_figure(encoded_image, cell_props, mask_data, show_annotations)

        # Create the graph object
        visualization = dcc.Graph(
            id="cell-visualization-graph",
            figure=fig,
            config={"displayModeBar": False, "staticPlot": False},
            style={"width": "100%", "height": "100%"},
        )

        # Success status
        status = dbc.Alert(
            f"Processed image: {len(cell_props)} cells detected in {total_processing_time:.2f} seconds",
            color="success",
            dismissable=True,
            is_open=True,
        )

        return cell_props, mask_data, summary_content, visualization, cropped_images, status

    except Exception as e:
        print(f"Error processing image: {e!s}")

        # Error status
        error_status = dbc.Alert(f"Error processing image: {e!s}", color="danger", dismissable=True, is_open=True)

        return None, None, html.P(f"Error processing image: {e!s}"), dash.no_update, None, error_status


# Callback to show cell details when clicked
@app.callback(
    [Output("selected-cell-image", "children"), Output("selected-cell-details", "children")],
    Input("cell-visualization-graph", "clickData"),
    [
        State("cell-data-store", "data"),
        State("processed-image-store", "data"),
        State("mask-data-store", "data"),
        State("image-hash-store", "data"),
        State("cropped-images-store", "data"),
    ],
    prevent_initial_call=True,
)
def display_selected_cell(click_data, cell_data, encoded_image, mask_data, img_hash, cropped_images):
    if not click_data or not cell_data or not mask_data:
        return html.Div(), html.P("Process an image and then click on a cell to view details", className="text-muted")

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

            if not cell:
                return html.Div(), html.P(f"Cell data not found for ID {cell_id}", className="text-muted")

        # Check if we have a cached cropped image
        cell_id = cell["id"]
        cropped_img_base64 = None

        if cropped_images and str(cell_id) in cropped_images:
            # Use cached cropped image
            cropped_img_base64 = cropped_images[str(cell_id)]
        elif img_hash:
            # Try to get from disk cache
            cropped_img_base64 = get_cropped_image(img_hash, cell_id)

        # Determine border color based on prediction or size
        predicted_props = cell.get("predicted_properties", {})
        if "concentration" in predicted_props:
            concentration = predicted_props.get("concentration", 0)
            border_color = get_concentration_color(concentration)
        else:
            border_color = "green" if cell["is_large"] else "red"

        if cropped_img_base64:
            # Use the cached cropped image directly
            cell_image = html.Img(
                src=cropped_img_base64,
                style={"width": "100%", "border": f"3px solid {border_color}"},
            )
        else:
            # Create the cropped image dynamically (fallback)
            y0, x0, y1, x1 = cell["bbox"]
            padding = 10
            y0 = max(0, y0 - padding)
            x0 = max(0, x0 - padding)
            y1 = min(len(mask_data["mask"]), y1 + padding)
            x1 = min(len(mask_data["mask"][0]), x1 + padding)

            x1 - x0
            y1 - y0

            # Create a zoomed-in view of the cell
            cell_fig = go.Figure()

            # Get dimensions of the full image
            content_type, content_string = encoded_image.split(",")
            decoded = base64.b64decode(content_string)
            img = np.array(Image.open(io.BytesIO(decoded)))
            full_height, full_width = img.shape[:2]

            # Add the original image
            cell_fig.add_layout_image(
                {
                    "source": encoded_image,
                    "xref": "x",
                    "yref": "y",
                    "x": 0,
                    "y": 0,
                    "sizex": full_width,
                    "sizey": full_height,
                    "sizing": "stretch",
                    "opacity": 1,
                    "layer": "below",
                }
            )

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

        # Create cell details with predicted properties
        cell_details = [
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
        cell_details.extend(
            [
                html.H6("Cell Metadata:", className="mt-3 mb-2"),
                html.P(f"Area: {int(cell['area'])} pixels"),
                html.P(f"Perimeter: {cell['perimeter']:.2f} pixels"),
                html.P(f"Eccentricity: {cell['eccentricity']:.3f}"),
                html.P(f"Centroid: ({cell['centroid_x']:.1f}, {cell['centroid_y']:.1f})"),
            ]
        )

        # Add size classification for non-classified cells
        if "predicted_class" not in predicted_props:
            is_large = cell["is_large"]
            cell_details.append(
                html.P(
                    [
                        "Size Classification: ",
                        html.Span(
                            "Large" if is_large else "Small",
                            style={"color": "green" if is_large else "red", "fontWeight": "bold"},
                        ),
                    ]
                )
            )

        return cell_image, cell_details

    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        print(f"Error in display_selected_cell: {e!s}")
        print(traceback_str)
        return html.Div(), html.P(f"Error: {e!s}", className="text-danger")


# Callback to update status after actions complete - Fixed unused arguments (ARG001)
@app.callback(
    Output("status-alert", "children", allow_duplicate=True),
    [Input("cell-data-store", "data"), Input("selected-cell-details", "children")],
    prevent_initial_call=True,
)
def clear_status_after_action(_cell_data, _cell_details):
    # This callback will fire after major actions complete
    # We return None to clear any previous status messages
    return None


def main():
    parser = argparse.ArgumentParser(description="Cell Analysis App with Lactase Classification")
    parser.add_argument("--model_path", type=str, default=None, help="Path to ONNX classification model")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Load classification model if provided
    if args.model_path:
        success = load_classification_model(args.model_path)
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
