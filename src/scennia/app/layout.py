import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from scennia.app.figure import update_image_analysis_figure_layout

IMAGE_ANALYSIS_GRAPH_ID = "image-analysis-graph"
IMAGE_ANALYSIS_SEGMENTATION_ID = "image-analysis-segmentation"
IMAGE_ANALYSIS_CLASSIFICATION_ID = "image-analysis-classification"

STATISTICS_BODY_ID = "statistics-body"
STATISTICS_CELL_COUNT_ID = "statistics-cell-count"
STATISTICS_CELL_AREA_ID = "statistics-cell-area"
STATISTICS_FITER_ID = "statistics-filter"
STATISTICS_FITER_RESET_ID = "statistics-filter-reset"

CELL_INFO_BODY_ID = "cell-info-body"

HASH_STORE = "hash-store"
PROCESSED_HASH_STORE_ID = "processed-hash-store"
IMAGE_ANALYSIS_FILTER_STORE = "image-analysis-filter-store"
SELECTED_CELL_STORE = "selected-cell-store"


cell_info_placeholder = html.P(
    "Upload an image and then click on a cell to view details",
    className="text-muted m-0",
)
cell_info_processed_placeholder = html.P(
    "Click on a cell to view details",
    className="text-muted m-0",
)
statistics_placeholder = html.P(
    "Click or upload an image to view the statistics",
    className="text-muted m-0",
)


# App layout
def create_layout(show_image_upload=True):
    persistence_type = "session"

    title = [
        html.H1("SCENNIA: Prototype Image Analysis Platform", className="my-2 text-center"),
        html.P(
            "A prototype web app of an AI-powered image analysis platform for cultivated "
            "meat cell line development as part of the SCENNIA project, funded by the Bezos Earth Fund.",
            className="text-center",
        ),
    ]
    prepared_images_card = dbc.Card(
        className="mh-100 h-100",
        children=[
            dbc.CardHeader(
                children=dbc.Row(
                    children=[
                        dbc.Col("Images", className="col-auto"),
                        dbc.Col(id="prepared-image-count", className="col-auto me-auto"),
                        dbc.Col(
                            className="col-auto",
                            children=dbc.Button(
                                "Refresh Images",
                                id="refresh-prepared-images",
                                color="link",
                                style={
                                    "--bs-btn-padding-y": "0",
                                    "--bs-btn-padding-x": "0",
                                    "--bs-btn-border-width": "0",
                                    "vertical-align": "baseline",
                                },
                            ),
                        ),
                    ],
                ),
            ),
            dbc.CardBody(
                className="h-100 overflow-x-scroll",
                children=[
                    dcc.Loading(
                        id="loading-image",
                        type="circle",
                        show_initially=True,
                        className="mh-100",  # Align spinner by setting height to 100% even with no content
                        children=html.Div(
                            className="radio-group h-100",
                            children=dbc.RadioItems(
                                id="prepared-images",
                                name="images",
                                className="btn-group",
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary",
                                labelCheckedClassName="active",
                                label_style={
                                    "width": "150px",
                                    "padding-left": "6px",
                                    "padding-right": "6px",
                                },
                                persistence=bool(persistence_type),
                                persisted_props=["value"],
                                persistence_type=persistence_type,
                                options=[],
                            ),
                        ),
                    ),
                ],
            ),
        ],
    )
    image_upload_card = dbc.Card(
        className="mh-100 h-100",
        children=[
            dbc.CardHeader(
                children=dbc.Row([
                    dbc.Col("Image Upload", className="col-auto me-auto"),
                    dbc.Col(id="image-filename", className="col-auto"),
                ])
            ),
            dbc.CardBody([
                dcc.Upload(
                    html.Div(["Drag and Drop or ", html.A("Select an Image", className="link-primary")]),
                    id="upload-image",
                    style={
                        "width": "100%",
                        "height": "130px",
                        "lineHeight": "130px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                    multiple=False,
                    accept="image/*",  # Accept only image files
                ),
            ]),
        ],
    )
    image_analysis_card = dbc.Card(
        className="mh-100 h-100",
        children=[
            dbc.CardHeader(
                children=dbc.Row(
                    children=[
                        dbc.Col(className="col-auto", children="Image Analysis"),
                        dbc.Col(id="actual-lactate-concentration", className="col-auto me-auto"),
                        dbc.Col(
                            className="col-auto",
                            children=dbc.Switch(
                                id=IMAGE_ANALYSIS_SEGMENTATION_ID,
                                label="Show Segmentation",
                                value=True,
                                persistence=bool(persistence_type),
                                persisted_props=["value"],
                                persistence_type=persistence_type,
                                className="mb-0",
                                label_class_name="mb-0",
                            ),
                        ),
                        dbc.Col(
                            className="col-auto",
                            children=dbc.Switch(
                                id=IMAGE_ANALYSIS_CLASSIFICATION_ID,
                                label="Show Classification",
                                value=True,
                                persistence=bool(persistence_type),
                                persisted_props=["value"],
                                persistence_type=persistence_type,
                                className="mb-0",
                                label_class_name="mb-0",
                            ),
                        ),
                    ],
                ),
            ),
            dbc.CardBody(
                dcc.Loading(
                    id="loading-image-analysis",
                    type="circle",
                    show_initially=False,
                    overlay_style={
                        "visibility": "visible",
                        "filter": "blur(3px) opacity(70%)",
                    },
                    className="mh-100",  # Align spinner by setting height to 100% even with no content
                    children=[
                        dcc.Graph(
                            id=IMAGE_ANALYSIS_GRAPH_ID,
                            figure=update_image_analysis_figure_layout(go.Figure(), 0, 0, False),
                            config={
                                "displayModeBar": False,
                                "staticPlot": False,
                                "autosizable": True,
                                "scrollZoom": False,
                                "doubleClick": False,
                                "showTips": False,
                            },
                            style={
                                "width": "100%",
                                "height": "100%",
                                # Set minimum width and height to prevent page from jumping around.
                                "min-width": "800px",
                                "min-height": "650px",
                            },
                            responsive=False,  # Disabled: shrinks the image.
                        ),
                    ],
                ),
            ),
        ],
    )
    cell_info_card = dbc.Card(
        className="mh-100 h-100",
        children=[
            dbc.CardHeader(
                children=dbc.Row(
                    children=[
                        dbc.Col(className="col-auto", children="Cell Info"),
                        dbc.Col(id="cell-lactate-concentration", className="col-auto me-auto"),
                        dbc.Col(id="cell-id", className="col-auto"),
                    ],
                ),
            ),
            dbc.CardBody([
                dcc.Loading(
                    id="loading-cell-info",
                    type="circle",
                    show_initially=False,
                    delay_show=1000,  # Delay showing spinner since loading is usually fast
                    className="mh-100",  # Align spinner by setting height to 100% even with no content
                    overlay_style={
                        "visibility": "visible",
                        "filter": "blur(3px) opacity(25%)",
                    },
                    children=[html.Div(id=CELL_INFO_BODY_ID, children=cell_info_placeholder)],
                ),
            ]),
        ],
    )
    statistics_card = dbc.Card(
        children=[
            dbc.CardHeader(
                children=dbc.Row(
                    children=[
                        dbc.Col(className="col-auto", children="Statistics"),
                        dbc.Col(id=STATISTICS_CELL_COUNT_ID, className="col-auto"),
                        dbc.Col(id=STATISTICS_CELL_AREA_ID, className="col-auto me-auto"),
                        dbc.Col(id=STATISTICS_FITER_ID, className="col-auto"),
                        dbc.Col(
                            className="col-auto",
                            children=dbc.Button(
                                "Reset Filter",
                                id=STATISTICS_FITER_RESET_ID,
                                color="link",
                                disabled=True,
                                style={
                                    "--bs-btn-padding-y": "0",
                                    "--bs-btn-padding-x": "0",
                                    "--bs-btn-border-width": "0",
                                    "vertical-align": "baseline",
                                },
                            ),
                        ),
                    ],
                ),
            ),
            dbc.CardBody(
                dcc.Loading(
                    id="loading-statistics",
                    type="circle",
                    show_initially=False,
                    className="mh-100",  # Align spinner by setting height to 100% even with no content
                    children=[html.Div(id=STATISTICS_BODY_ID, children=statistics_placeholder)],
                ),
                style={
                    # Set minimum height to prevent page from jumping around.
                    "min-height": "100px",
                },
            ),
        ],
    )

    # Create main columns
    main_cols = [
        dbc.Col(width=8 if show_image_upload else 12, children=[prepared_images_card]),
    ]
    if show_image_upload:
        main_cols.append(dbc.Col(width=4, children=[image_upload_card]))
    main_cols.extend([
        dbc.Col(width=8, children=[image_analysis_card]),
        dbc.Col(width=4, children=[cell_info_card]),
        dbc.Col(width=12, children=[statistics_card]),
    ])

    return dbc.Container(
        fluid=True,
        style={
            "min-width": "1250px",
            "max-width": "1400px",
        },
        children=[
            # Row: Title
            dbc.Row(className="row-cols-1 g-2", children=[dbc.Col(title)]),
            # Row: Main content
            dbc.Row(
                className="row-cols-2 g-2 mb-2",
                children=main_cols,
            ),
            # Row: Footer
            dbc.Row(
                className="row-cols-1 g-2",
                children=[
                    dbc.Col(id="model-status", width=12),
                    dbc.Col(id="status-alert", width=12),
                ],
            ),
            # Hidden data stores
            dcc.Store(id=HASH_STORE, storage_type=persistence_type),
            dcc.Store(id=PROCESSED_HASH_STORE_ID, storage_type=persistence_type),
            dcc.Store(id=IMAGE_ANALYSIS_FILTER_STORE, storage_type=persistence_type),
            dcc.Store(id=SELECTED_CELL_STORE, storage_type=persistence_type),
        ],
    )
