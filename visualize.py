#!/usr/bin/env python3

import argparse
from collections import defaultdict
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import json
import logging
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from skimage import io
import tarfile


def read_csv(fp):
    logger = logging.getLogger()
    logger.info(f"Reading in {fp}")
    df = pd.read_csv(fp)
    logger.info(f"Read in {df.shape[0]:,} rows and {df.shape[1]:,} columns")

    # Check for the index values for each image and combination of parameters

    # Each table _must_ have a params_ix column
    assert "params_ix" in df.columns.values, f"Did not find column `params_ix` in {fp}"

    # `params_ix` must be an integer
    df = df.assign(params_ix = df.params_ix.apply(int))

    # If `image_ix` is present
    if 'image_ix' in df.columns.values:
    
        # `image_ix` must be an integer
        df = df.assign(image_ix = df.image_ix.apply(int))

    return df


def add_parameters(parameters, df):
    """
    For a table with a `params_ix` columns, add in the appropriate
    values for each variable from the table at self.parameters.
    """

    # Each table _must_ have a params_ix column
    assert "params_ix" in df.columns.values, f"Did not find column `params_ix`"

    # Add to the table
    return df.assign(
        **{
            col_name: df.params_ix.apply(
                col_values.get
            )
            for col_name, col_values in parameters.set_index("params_ix").items()
            if col_values.unique().shape[0] > 1
        }
    )

def read_images(fp):

    logger = logging.getLogger()

    # Extract the archive, if it hasn't been already
    folder = f"{fp}.extracted"
    if not os.path.exists(folder):
        os.mkdir(folder)
        logger.info(f"Decompressing {fp} into {folder}")
        with tarfile.open(fp, "r:gz") as tar:
            tar.extractall(folder)

    # Keep track of the path for each image, keyed by image and parameters
    images = defaultdict(dict)
    
    # Iterate over each image
    for fn in os.listdir(folder):

        # Parse the image index and parameter index from the file name
        image_ix, params_ix, _ = fn.split(".", 2)
        image_ix = int(image_ix)
        params_ix = int(params_ix)

        # Key the images first by image index, and then by params
        images[image_ix][params_ix] = os.path.join(folder, fn)

    return images


def launch_qvc(
    images=None,
    features=None,
    summary=None,
    parameters=None,
    host="0.0.0.0",
    port=8000,
    debug=False
):
    """Visualize a set of results from the quantify-view-cells pipeline."""

    # Get the logger
    logger = logging.getLogger()

    # Validate the input data
    for fp, label in [(images, 'images'), (features, 'features'), (summary, 'summary')]:
        assert os.path.exists(fp), f"Must provide valid path to {label}, run with --help for details"

    # Read in the data for all tables
    features = read_csv(features)
    summary = read_csv(summary)
    parameters = read_csv(parameters)

    # Expand the `params_ix` column from `features` and `summary`  to
    # include the actual values from `parameters`
    features = add_parameters(parameters, features)
    summary = add_parameters(parameters, summary)

    # Read in all of the images
    images = read_images(images)

    # Set up the app
    logger.info("Initializing the VizApp object")
    app = VizApp(features, summary, parameters, images)

    # Launch it
    logger.info("Launching the server")
    app.run_server(
        host=host,
        port=port,
        debug=debug
    )


class CallbackResponse:
    """Object to help manage a complex multi-output callback response."""

    def __init__(self, output_functions, default_values):

        # Values are formatted as a dict
        assert isinstance(default_values, dict)

        # Save the default values
        self.values = default_values

        # Functions are gathered in a list
        assert isinstance(output_functions, list)

        # Save the output functions
        self.output_functions = output_functions

    def set(self, **kwargs):
        """Set one or more of the values."""

        for k, v in kwargs.items():

            assert k in self.values, f"Key not found ({k})"

            self.values[k] = v

    def resolve(self, **kwargs):
        """Return the output data for the callback."""

        self.set(**kwargs)

        return [
            f(self.values)
            for f in self.output_functions
        ]


class VizApp:

    def __init__(
        self,
        features,
        summary,
        parameters,
        images,
        theme=dbc.themes.FLATLY,
        title="Quantify View Cells (QVC)",
        suppress_callback_exceptions=True
    ):

        # Get the logger
        self.logger = logging.getLogger()

        # Attach the data
        self.features = features
        self.summary = summary
        self.parameters = parameters
        self.images = images

        # Create an indexed list of all possible parameters used
        self.index_possible_parameters()

        # Make a list of all columns in the feature data which are numeric
        self.numeric_cols = [
            col_name
            for col_name, col_values in self.features.items()
            if is_numeric(col_values) and col_name not in ["image_ix", "params_ix"]
        ]

        # Sort the images by image and params index
        self.summary.sort_values(
            by=["image_ix", "params_ix"],
            inplace=True
        )

        # Set up the type of plots which are available
        self.plot_types = dict(
            scatter=dict(
                label="Scatter"
            ),
            heatmap=dict(
                label="Heatmap"
            ),
            contour=dict(
                label="Contour"
            )
        )

        # Set up the options (ids and labels) for formatting the figures
        self.data_display_options = dict(
            x="X-axis",
            y="Y-axis",
            color="Color"
        )

        # Create the Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[theme]
        )

        # Set the title of the app
        self.app.title = title

        # Suppress callback exceptions by default
        self.app.config.suppress_callback_exceptions = suppress_callback_exceptions

        # Set up the layout
        self.app.layout = self.layout

        # Attach the callbacks
        self.decorate_callbacks()

    def index_possible_parameters(self):
        """Create an indexed list of all possible parameters for each."""

        self.parameter_index = {
            param_col: dict(enumerate(param_values.drop_duplicates().sort_values().values))
            for param_col, param_values in self.parameters.items()
            if param_col not in ["params_ix"] and param_values.unique().shape[0] > 1
        }

    def layout(self):
        """Set up the layout of the app."""
        return dbc.Container(
            [
                # Format the navbar at the top of the page
                self.navbar(),
                # All elements are grouped into tabs
                dcc.Tabs(
                    # Set the ID for the tabbed page
                    id="qvc-tabs",
                    # Show the image tab by default
                    value="image-tab",
                    # Set up the contents of the tabs
                    children=[
                        # The first tab is the image display
                        dcc.Tab(
                            label="Image Display",
                            value="image-tab",
                            children=self.layout_image_display()
                        ),
                        # The second tab is the image summary table
                        dcc.Tab(
                            label="Image Metrics",
                            value="image-metrics-tab",
                            children=self.layout_data_display("summary")
                        ),
                        # The third tab is the feature summary table
                        dcc.Tab(
                            label="Feature Metrics",
                            value="feature-metrics-tab",
                            children=self.layout_data_display("features")
                        )
                    ]
                )
            ]
        )

    def navbar(self):
        """Format the navbar at the top of the page."""
        return dbc.NavbarSimple(
            brand="Quantify & View Cells",
            color="primary",
            dark=True
        )

    def layout_image_display(self):

        # Initialize the display with the first image
        img = io.imread(self.images[1][0])

        # Get the list of valid image indices
        img_keys = pd.Series(self.images.keys()).sort_values().values
        
        return dbc.Row(
            [
                dbc.Col(
                    dbc.Form(
                        [
                            dbc.FormGroup([
                                dbc.Label("Show Image (#)"),
                                dcc.Slider(
                                    id='image-display-selector',
                                    value=img_keys[0],
                                    min=img_keys[0],
                                    max=img_keys[-1],
                                    marks={
                                        str(i): str(i)
                                        for i in img_keys
                                    },
                                    included=False,
                                )
                            ])
                        ] \
                            # Show a slider for each of the parameters used
                            + self.format_param_slider_list()
                    ),
                    # Relative width of the menu column
                    width=4,
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=px.imshow(img),
                        id='image-display'
                    ),
                    width=8
                )
            ],
            justify="center",
            align="center",
            className="h-50",
        )

    def format_param_slider_list(self):
        """Render a list of sliders to select from each parameter."""

        return [
            dbc.FormGroup(
                [
                    dbc.Label(f"Select {param_col}"),
                    dcc.Slider(
                        id=f"image-selector-{param_col}",
                        value=0,
                        min=0,
                        max=len(param_values) - 1,
                        step=1,
                        marks={
                            str(k): str(v)
                            for k, v in param_values.items()
                        },
                        included=False,
                    )
                ]
            )
            for param_col, param_values in self.parameter_index.items()
        ]

    def infer_param_ix(self, selected_params):
        """
        Input is a dict with the _index_ of each param that was selected.
        Output is the `params_ix` which corresponds to that set of values
        """

        # Start with the complete set of parameter combinations
        df = self.parameters

        # Iterate over each param
        for param_key, param_ix in selected_params.items():

            # Get the value from this index
            param_value = self.parameter_index[param_key][param_ix]

            # Filter the table
            df = df.query(f"{param_key} == {param_value}")

            # Make sure that we have some parameters left
            msg = f"No parameter found with {param_key} == {param_value}"
            assert df.shape[0] > 0, msg

        # After filtering, there should only be one remaining
        msg = f"Incomplete parameter filtering: ({json.dumps(selected_params)})"
        assert df.shape[0] == 1, msg

        return df["params_ix"].values[0]


    def layout_data_display(self, data_type):
        """
        Display data based on the `data_type` provided, either 'features' or 'images'
        """

        return dbc.Container(
            [
                dbc.Row(
                    [
                        # Selectors used to format the display
                        dbc.Col(
                            self.format_data_display(data_type),
                            width=4
                        ),
                        # The actual data display element
                        dbc.Col(
                            self.render_data_display(data_type),
                            width=8
                        )
                    ]
                )
            ]
        )

    def format_data_display(self, data_type):
        """
        Render the selectors used to drive a data display.
        """

        return dbc.Form(
            [
                # PLOT TYPE
                dbc.FormGroup(
                    [
                        dbc.Label("Plot Type"),
                        dcc.Dropdown(
                            id=dict(
                                elem="data-selector",
                                data_type=data_type,
                                input_id="plot-type"
                            ),
                            options=[
                                dict(
                                    label=v["label"],
                                    value=k
                                )
                                for k, v in self.plot_types.items()
                            ],
                            value=list(self.plot_types.keys())[0]
                        )
                    ]
                )
            ] + [
                # X, Y, and color
                dbc.FormGroup(
                    [
                        dbc.Label(input_label),
                        dcc.Dropdown(
                            id=dict(
                                elem="data-selector",
                                data_type=data_type,
                                input_id=input_id
                            ),
                            options=[
                                dict(
                                    label=col_name,
                                    value=col_name
                                )
                                for col_name in self.numeric_cols
                            ],
                            value=self.numeric_cols[i]
                        )
                    ]
                )
                for i, [input_id, input_label] in enumerate(self.data_display_options.items())
            ] + [
                # FILTER BY PARAMETER
                dbc.Collapse(
                    id=dict(
                        elem_data="selector-collapse",
                        data_type=data_type,
                        filter_param=param_key
                    ),
                    children=dbc.FormGroup(
                        [
                            dbc.Label(param_key),
                            dcc.Dropdown(
                                id=dict(
                                    elem="data-selector",
                                    data_type=data_type,
                                    filter_param=param_key
                                ),
                                options=[
                                    dict(
                                        label=val_str,
                                        value=val_ix,
                                    )
                                    for val_ix, val_str in param_values.items()
                                ],
                                value=0
                            )
                        ]
                    ),
                    is_open=True
                )
                for param_key, param_values in self.parameter_index.items()
            ],
            style=dict(
                marginTop="10px"
            )
        )

    def render_data_display(self, data_type):
        """Placeholder to render the data display."""

        return dcc.Graph(
            id=dict(
                elem="data-display",
                data_type=data_type
            )
        )

    def layout_filter_data(self):

        initial_column = self.numeric_cols[0]

        return [
            dbc.Row(
                [
                    dbc.Col(
                        "Filter Data By:"
                    ),
                    dbc.Col(
                        "Minimum Value:"
                    ),
                    dbc.Col(
                        "Maximum Value:"
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="filter-data-by",
                            options=[
                                {'label': i, 'value': i}
                                for i in self.numeric_cols
                            ],
                            value=initial_column,
                            multi=False
                        )
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="filter-min-value",
                            type="number",
                            min=self.features[initial_column].min(),
                            max=self.features[initial_column].max(),
                            value=self.features[initial_column].min(),
                        )
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="filter-max-value",
                            type="number",
                            min=self.features[initial_column].min(),
                            max=self.features[initial_column].max(),
                            value=self.features[initial_column].max(),
                        )
                    ),
                ]
            )
        ]

    def layout_summary_figure(self):
        return dbc.Row(
            [
                dbc.Col(
                    self.layout_figure_options(),
                    width=4
                ),
                dbc.Col(
                    self.layout_figure_display(),
                    width=8
                )
            ] + self.layout_filter_data()
        )

    def layout_figure_options(self):
        return [
            html.Br(),
            "Display Type",
            html.Br(),
            dcc.Dropdown(
                id="display-type",
                options=[
                    {"label": i.title(), "value": i}
                    for i in [
                        "scatter",
                        "heatmap",
                        "contour",
                    ]
                ],
                value="scatter"
            ),
            html.Br(),
            "X-axis",
            html.Br(),
            dcc.Dropdown(
                id="display-x",
                options=[
                    {"label": i, "value": i}
                    for i in self.numeric_cols
                ],
                value=self.numeric_cols[0]
            ),
            html.Br(),
            "Y-axis",
            html.Br(),
            dcc.Dropdown(
                id="display-y",
                options=[
                    {"label": i, "value": i}
                    for i in self.numeric_cols
                ],
                value=self.numeric_cols[1]
            )
        ]

    def feature_display(
        self,
        plot_params
    ):

        self.logger.info(json.dumps(plot_params, indent=3))

        # Set up the data to be used for plotting
        if plot_params["data_type"] == "summary":
            plot_df = self.summary
        else:
            assert plot_params["data_type"] == "features", plot_params["data_type"]
            plot_df = self.features

        # Check each of the parameters which have been specified
        for param_key, param_value_i in plot_params["filter_param"].items():

            # If this parameter is not being used in the display
            if param_key not in plot_params["input_id"].values():

                # Get the selected value of the parameter to show
                param_value = self.parameter_index[param_key][param_value_i]
                self.logger.info(f"Filtering {param_key} to {param_value}")
                plot_df = plot_df.query(f"{param_key} == {param_value}")
                assert plot_df.shape[0] > 0, "No data to display"

        # Define the function which will be used
        plotly_f, plotly_kwargs = dict(
            scatter = (px.scatter, ["x", "y", "color"]),
            heatmap = (px.density_heatmap, ["x", "y"]),
            contour = (px.density_contour, ["x", "y"]),
        )[plot_params["input_id"]["plot-type"]]

        # Set up the figure
        fig = plotly_f(
            plot_df,
            **{
                k: plot_params["input_id"][k]
                for k in plotly_kwargs
            }
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    def decorate_callbacks(self):
        """Define the interactivity of the app"""
        
        # Show the selected image
        @self.app.callback(
            Output("image-display", "figure"),
            [
                Input("image-display-selector", "value")
            ] + [
                Input(f"image-selector-{param_col}", "value")
                for param_col in self.parameter_index.keys()
            ]
        )
        def show_selected_image(*input_args):

            # Parse the selected image index
            image_ix = input_args[0]

            # Get the list of parameter values
            selected_params = dict(
                zip(
                    self.parameter_index.keys(),
                    input_args[1:]
                )
            )

            # Get the index of this combination of parameters
            param_ix = self.infer_param_ix(selected_params)

            # Read the image from the file
            img = io.imread(self.images[image_ix][param_ix])

            # Show that image
            return px.imshow(img)

        # Show / hide the collapse for the parameter menus
        @self.app.callback(
            Output(
                dict(
                    elem_data="selector-collapse",
                    data_type=MATCH,
                    filter_param=ALL
                ),
                "is_open"
            ),
            [
                Input(
                    dict(
                        elem="data-selector",
                        data_type=MATCH,
                        input_id=ALL
                    ),
                    "value"
                )
            ]
        )
        def show_hide_filter_param_collapse(input_list):
            ctx = dash.callback_context

            # Close the collapse if the param was selected
            # as a continuous element in the display
            return [
                elem["id"]["filter_param"] not in input_list
                for elem in ctx.outputs_list
            ]

        # Render each of the data graphs
        @self.app.callback(
            Output(
                dict(
                    elem="data-display",
                    data_type=MATCH
                ),
                "figure"
            ),
            [
                Input(
                    dict(
                        elem="data-selector",
                        data_type=MATCH,
                        input_id=ALL
                    ),
                    "value"
                ),
                Input(
                    dict(
                        elem="data-selector",
                        data_type=MATCH,
                        filter_param=ALL
                    ),
                    "value"
                )
            ]
        )
        def render_data_figure(*_):
            # Get the callback context
            ctx = dash.callback_context

            # Format the input arguments
            plot_params = {
                input_type: {
                    i["id"][input_type]: i["value"]
                    for i in input_elems
                }
                for input_type, input_elems in zip(
                    ["input_id", "filter_param"],
                    ctx.inputs_list
                )
            }

            # Get the plot type as well
            plot_params["data_type"] = ctx.inputs_list[0][0]['id']['data_type']

            # Return a figure based on these params
            return self.feature_display(
                plot_params
            )

        # Update the min and max for the filter column
        @self.app.callback(
            [
                Output("filter-min-value", "min"),
                Output("filter-min-value", "max"),
                Output("filter-min-value", "value"),
                Output("filter-max-value", "min"),
                Output("filter-max-value", "max"),
                Output("filter-max-value", "value"),
            ],
            [
                Input("filter-data-by", "value")
            ]
        )
        def update_filter_min_max(col_name):

            response = CallbackResponse(
                [
                    lambda d: d['values'].min(),
                    lambda d: d['values'].max(),
                    lambda d: d['values'].min(),
                    lambda d: d['values'].min(),
                    lambda d: d['values'].max(),
                    lambda d: d['values'].max(),
                ],
                dict(
                    values=None
                )
            )

            return response.resolve(
                values=self.features[col_name]
            )

        # Update the figure display
        @self.app.callback(
            Output("figure-display", "figure"),
            [
                Input("display-type", "value"),
                Input("display-x", "value"),
                Input("display-y", "value"),
                Input("filter-data-by", "value"),
                Input("filter-min-value", "value"),
                Input("filter-max-value", "value"),
            ]
        )
        def update_figure_display(
            figure_type,
            x_col,
            y_col,
            filter_col,
            filter_min,
            filter_max
        ):
            return self.feature_display(
                figure_type,
                x_col,
                y_col,
                filter_col,
                filter_min,
                filter_max
            )

    def run_server(
        self,
        host="0.0.0.0",
        port=8000,
        debug=False
    ):
        """Launch and serve the app."""
        self.app.run_server(
            host=host,
            port=port,
            debug=debug,
        )

def select_sigfigs(df):
    """Pick the number of decimals to round each columns to."""

    # Output will be a dict
    decimals = dict()

    # For each column
    for col_name, col_values in df.items():

        # Get the median value
        median_value = col_values.median()

        # Get the order of magnitude
        om = int(np.log10(median_value))

        # If the number is > 1,000
        if om > 3:

            # Round off to numbers
            decimals[col_name] = 0

        # Otherwise
        else:

            # Provide 4 sig figs
            decimals[col_name] = -1 * (om - 4)
    
    return decimals

# Function to test if all values in a Series are numeric
def is_numeric(v):
    return v.apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).notnull(
    ).all()

if __name__ == "__main__":

    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [QVC] %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Write logs to STDOUT
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(
        description="""
Launch Quantify-View-Cells (QVC) Visualization

Usage:

    python3 visualize.py \
        --images output_images.tar.gz \
        --features feature.info.csv.gz \
        --summary image.summary.csv.gz

        """
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Compressed archive of output images"
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Compressed CSV with feature-level data"
    )

    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Compressed CSV with image-level summaries of feature data"
    )

    parser.add_argument(
        "--parameters",
        type=str,
        required=True,
        help="Table of parameters used for each analysis (CSV)"
    )

    parser.add_argument(
        '--host',
        type=str,
        default="0.0.0.0",
        help='Address to host GLAM Browser'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to host GLAM Browser'
    )
    parser.add_argument(
        '--debug',
        action="store_true",
        help='Run in debug mode'
    )

    args = parser.parse_args()

    launch_qvc(
        **args.__dict__
    )