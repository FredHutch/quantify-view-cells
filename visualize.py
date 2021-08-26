#!/usr/bin/env python3

import argparse
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
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
    return df


def read_images(fp):

    logger = logging.getLogger()

    # Extract the archive, if it hasn't been already
    folder = f"{fp}.extracted"
    if not os.path.exists(folder):
        os.mkdir(folder)
        logger.info(f"Decompressing {fp} into {folder}")
        with tarfile.open(fp, "r:gz") as tar:
            tar.extractall(folder)

    # Keep track of the path for each image
    images = {
        int(fn.split(".", 1)[0]): os.path.join(folder, fn)
        for fn in os.listdir(folder)
    }

    return images


def launch_qvc(
    images=None,
    features=None,
    summary=None,
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

    # Read in the data for both tables
    features = read_csv(features)
    summary = read_csv(summary)

    # Read in all of the images
    images = read_images(images)

    # Set up the app
    logger.info("Initializing the VizApp object")
    app = VizApp(features, summary, images)

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
        images,
        theme=dbc.themes.FLATLY,
        title="Quantify View Cells (QVC)",
        suppress_callback_exceptions=True
    ):

        # Get the logger
        self.logger = logging.getLogger()

        # Create the Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[theme]
        )

        # Set the title of the app
        self.app.title = title

        # Suppress callback exceptions by default
        self.app.config.suppress_callback_exceptions = suppress_callback_exceptions

        # Attach the data
        self.features = features
        self.summary = summary
        self.images = images

        # Make a list of all columns in the feature data which are numeric
        self.numeric_cols = [
            col_name
            for col_name, col_values in self.features.items()
            if is_numeric(col_values)
        ]

        # Sort the images by index
        self.summary.sort_values(
            by="ix",
            inplace=True
        )

        # Rename the `ix` column to `id`
        self.summary = self.summary.rename(
            columns=dict(ix="id")
        )

        # Set up the layout
        self.app.layout = self.layout

        # Attach the callbacks
        self.decorate_callbacks()

    def layout(self):
        """Set up the layout of the app."""
        return dbc.Container(
            [
                # The top element is the image display
                self.layout_image_display(),
                
                # Below that is a table summarizing each image
                self.layout_image_table(),

                html.Hr(),

                # Next we have a menu allowing the user to filter
                # the data points which are shown below
                *self.layout_filter_data(),

                # Finally, the bottom has an interactive
                # figure generation utility with flexible menus
                self.layout_summary_figure()
            ]
        )

    def layout_image_display(self):

        # Initialize the display with the first image
        img = io.imread(self.images[1])
        
        return dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        figure=px.imshow(img),
                        id='image-display'
                    ),
                    width=8
                ),
                dbc.Col(
                    [
                        "Show Image (#)",
                        html.Br(),
                        dcc.Dropdown(
                            id="image-display-selector",
                            options=[
                                {"label": i, "value": i}
                                for i in pd.Series(
                                    self.images.keys()
                                ).sort_values().values
                            ],
                            value=1,
                            multi=False,
                        )
                    ],
                    width=4
                )
            ],
            justify="center",
            align="center",
            className="h-50",
        )
        

    def layout_image_table(self):
        """Render a table with the image summary data."""

        return html.Div(
            [
                self.image_table_element(),
                *self.image_table_hide_columns(),
            ]
        )

    def image_table_hide_columns(self):

        return [
            dbc.Row(
                dbc.Col("Hide Columns:", width=4),
                style=dict(marginTop="5px")
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        id="image-table-hidden-column-selector",
                        options=[
                            {'label': i, 'value': i}
                            for i in self.summary.columns
                        ],
                        value=['id'],
                        multi=True
                    ),
                    width=4
                ),
                style=dict(marginBottom="5px")
            )
        ]

    def image_table_element(self):
        return dash_table.DataTable(
            id='image-table',
            columns=[
                {"name": i, "id": i}\
                for i in self.summary.columns
            ],
            sort_action='native',
            filter_action='native',
            row_selectable='multi',
            selected_rows=list(range(self.summary.shape[0])),
            sort_mode='multi',
            page_action='native',
            page_current= 0,
            page_size= 10,
            hidden_columns=["id"],
            data=self.summary.round(
                select_sigfigs(self.summary)
            ).to_dict(
                'records'
            ),
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            css=[{"selector": ".show-hide", "rule": "display: none"}]
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
            ]
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

    def layout_figure_display(self):

        initial_type = "scatter"
        initial_x = self.numeric_cols[0]
        initial_y = self.numeric_cols[1]

        filter_col = self.numeric_cols[0]
        filter_values = self.features[filter_col]
        filter_min = filter_values.min()
        filter_max = filter_values.max()
        
        return dcc.Graph(
            figure=self.feature_display(
                initial_type,
                initial_x,
                initial_y,
                filter_col,
                filter_min,
                filter_max,
            ),
            id="figure-display"
        )

    def feature_display(
        self,
        figure_type,
        x_col,
        y_col,
        filter_col,
        filter_min,
        filter_max
    ):

        # Define the function which will be used
        plotly_f = dict(
            scatter = px.scatter,
            heatmap = px.density_heatmap,
            contour = px.density_contour,
        )[figure_type]

        filtered_data = self.features.query(
            f"{filter_col} >= {filter_min}"
        ).query(
            f"{filter_col} <= {filter_max}"
        )

        msg = f"Filtering invalid: {filter_col} >= {filter_min}, <= {filter_max}"
        assert filtered_data.shape[0] > 0, msg

        fig = plotly_f(
            filtered_data,
            x=x_col,
            y=y_col,
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    def decorate_callbacks(self):
        """Define the interactivity of the app"""
        
        # Hide columns in the image summary table
        @self.app.callback(
            Output("image-table", "hidden_columns"),
            [
                Input("image-table-hidden-column-selector", "value"),
            ]
        )
        def hide_columns_image_table(col_list):
            return col_list

        # Show the selected image
        @self.app.callback(
            Output("image-display", "figure"),
            [
                Input("image-display-selector", "value")
            ]
        )
        def show_selected_image(ix):

            # Read the image from the file
            img = io.imread(self.images[ix])

            # Show that image
            return px.imshow(img)

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