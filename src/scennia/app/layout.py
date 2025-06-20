import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from scennia.app.image import update_image_analysis_figure_layout

cell_info_placeholder = html.P(
    "Upload an image and then click on a cell to view details",
    className="text-muted m-0",
)
cell_info_processed_placeholder = html.P(
    "Click on a cell to view details",
    className="text-muted m-0",
)
summary_placeholder = html.P(
    "Upload an image to view the summary",
    className="text-muted m-0",
)


# App layout
def create_layout():
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
            dbc.CardHeader("Images"),
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
                                id="show-segmentation",
                                label="Show Segmentation",
                                value=True,
                                className="mb-0",
                                label_class_name="mb-0",
                            ),
                        ),
                        dbc.Col(
                            className="col-auto",
                            children=dbc.Switch(
                                id="show-classification",
                                label="Show Classification",
                                value=True,
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
                        "filter": "blur(3px) opacity(25%)",
                    },
                    className="mh-100",  # Align spinner by setting height to 100% even with no content
                    children=[
                        dcc.Graph(
                            id="image-analysis",
                            figure=update_image_analysis_figure_layout(go.Figure(), 0, 0),
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
                                "min-height": "600px",
                            },
                            responsive=False,  # Disabled: shrinks the image.
                        ),
                    ],
                ),
            ),
        ],
    )
    summary_card = dbc.Card(
        children=[
            dbc.CardHeader(
                children=dbc.Row(
                    children=[
                        dbc.Col(className="col-auto", children="Summary"),
                        dbc.Col(id="detected-cell-count", className="col-auto"),
                        dbc.Col(id="median-cell-area", className="col-auto"),
                        dbc.Col(id="mean-cell-area", className="col-auto me-auto"),
                    ],
                ),
            ),
            dbc.CardBody(
                dcc.Loading(
                    id="loading-summary",
                    type="circle",
                    show_initially=False,
                    className="mh-100",  # Align spinner by setting height to 100% even with no content
                    children=[html.Div(id="summary", children=summary_placeholder)],
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
                    delay_show=3000,  # Delay showing spinner since loading is usually fast
                    className="mh-100",  # Align spinner by setting height to 100% even with no content
                    overlay_style={
                        "visibility": "visible",
                        "filter": "blur(3px) opacity(25%)",
                    },
                    children=[html.Div(id="cell-info", children=cell_info_placeholder)],
                ),
            ]),
        ],
    )
    model_status = html.Div(id="model-status")
    alerts = html.Div(id="status-alert")

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
                children=[
                    dbc.Col(width=8, children=[prepared_images_card]),
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
