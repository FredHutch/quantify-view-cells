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
import os
import pandas as pd
import tarfile


def read_csv(fp):
    logger = logging.getLogger()
    logger.info(f"Reading in {fp}")
    df = pd.read_csv(fp)
    logger.info(f"Read in {df.shape[0]:,} rows and {df.shape[1]:,} columns")
    return df


def read_images(fp):

    logger = logging.getLogger()
    logger.info(f"Reading in {fp}")
    with tarfile.open(fp, "r:gz") as tar:

        images = {
            filename: tar.extractfile(filename).read()
            for filename in tar.getnames()
        }
    logger.info(f"Read in {len(images):,} images")

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

                # Next we have a menu allowing the user to filter
                # the data points which are shown below
                self.layout_filter_data(),

                # Finally, the bottom has an interactive
                # figure generation utility with flexible menus
                self.layout_summary_figure()
            ]
        )

    def decorate_callbacks(self):
        """Define the interactivity of the app"""
        pass

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

    args = parser.parse_args()

    launch_qvc(
        **args.__dict__
    )