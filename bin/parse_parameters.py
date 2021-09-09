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
    '--parameters',
    type=str,
    required=True,
    help='Parameters used for execution in semi-colon delimited list'
)
parser.add_argument(
    '--czifile',
    type=str,
    required=True,
    help='String used for the input image filepath (czifile)'
)
parser.add_argument(
    '--output',
    type=str,
    required=True,
    help='Path to output file (txt), one set of parameters encoded on each line as JSON'
)

# Parse the arguments
args = parser.parse_args()

# Keep track of all possible parameters
params = dict()

# Split up the parameter string by semi-colon
for param_field in args.parameters.split(";"):

    # Skip empty fields
    if len(param_field) == 0:
        continue

    # Every field must have a key=value
    assert "=" in param_field, "Field must contain '=' (%s)" % param_field

    # Split up the key and the value
    key, value = param_field.split("=", 1)

    assert len(key) > 0, f"Field must be formatted as Key=Value ({param_field})"
    assert len(value) > 0, f"Field must be formatted as Key=Value ({param_field})"

    # Make sure that this key is not a duplicate
    assert key not in params, f"Duplicate key: {key}"

    # The value will be a list
    params[key] = list()

    # Iterate over multiple values as a comma-delimited list
    for single_value in value.split(","):

        # If this is an empty value
        if len(single_value) == 0:

            # Skip it
            continue

        # If this value is a range
        if '-' in single_value:

            # Parse the minimum and the maximum
            min_v, max_v = single_value.split("-", 1)

            # Each must be able to be coerced to a string
            try:
                min_v = int(min_v)
            except:
                raise Exception(f"Value must be an integer: {min_v}")
            try:
                max_v = int(max_v)
            except:
                raise Exception(f"Value must be an integer: {max_v}")

            # Add values for the whole range
            params[key].extend(list(range(min_v, max_v + 1)))

        # If this is not a range
        else:

            # Add the single value
            params[key].append(single_value)

# Remove any duplicate values for each param
params = {
    k: list(set(v))
    for k, v in params.items()
}

# Make sure that czifile wasn't specified by the user
assert "czifile" not in params, "Cannot include czifile in the --parameters"

# Now add 'czifile' to the params
params['czifile'] = [args.czifile]

# Write out all possible versions of the parameters
with open(args.output, "w") as handle:

    # Iterate over all possible combinations of values
    for value_comb in product(*params.values()):

        # Format the values as a dict with the appropriate key
        dat_comb = dict(zip(params.keys(), value_comb))

        # Write out the combination of parameters as JSON
        handle.write(json.dumps(dat_comb) + '\n')
