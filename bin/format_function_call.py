#!/usr/bin/env python3
import argparse
from itertools import product
import json

# Create the parser
parser = argparse.ArgumentParser(
    description='Parse parameters for quantify-view-cells'
)

# Add the arguments
parser.add_argument(
    '--parameter-json',
    type=str,
    required=True,
    help='Parameters used for execution of a single function call, in JSON format as a dict'
)
parser.add_argument(
    '--function-call',
    type=str,
    required=True,
    help='String used to run MATLAB'
)

# Parse the arguments
args = parser.parse_args()

# Read in the values
params = json.loads(args.parameter_json)

# Write out the formatted function call to STDOUT
print(args.function_call.format(**params))
