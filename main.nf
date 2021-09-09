#!/usr/bin/env nextflow

// Using DSL-2
nextflow.enable.dsl=2

// Set default parameters
params.help = false
params.function_call = false
params.parameters = false
params.assets = false
params.sample_sheet = false
params.output = false
params.module = ""
params.max_threads = 10000
params.output_csv = "output.csv"
params.output_img = "output.tif"
params.czifile = "input.czi"

// Function which prints help message text
def helpMessage() {
    log.info"""
Usage:

nextflow run FredHutch/quantify-view-cells <ARGUMENTS>

Required Arguments:
    --function_call         Formatted function call which will analyze an image
                            (named '{czifile}') and write an output (named 'output.mat').
                            The function call is formatted to include keyword arguments inside {} braces.
                            The only required parameter is "{czifile}", and all others must
                            be referenced with the --parameters flag as described below.
    --parameters            Any additional parameters to be used with the function call can
                            be specified with a semi-colon delimited list. For example, the function

                                "[data, dataT, O]=FunctionName('{czifile}',{idx},{paramA},{paramB});writetable(dataT, 'output.csv');imwrite(O,'output.tif','tiff')"
                            
                            Could be run with the following parameters:

                                "idx=1;paramA=10,20,30,40;paramB=40-50;"

                            With the effect of executing the analysis for each image with all possible
                            combinations of those variables. Note that commas indicate a list of specific
                            values, while a dash is used to specify all integers in that range (inclusive).

    --assets                Any additional files which should be present in the working
                            directory in order for the script to run.
                            Multiple files (or folders) can be specified with a comma-delimited
                            list, while also supporting wildcards. Double wildcards cross directory boundaries.
    --sample_sheet          List of images to process using the specified analysis.
                            The file must be in CSV format, and the path to the image must be
                            listed in a column named 'uri'.
    --output                Path to output directory

Optional Arguments:
    --module                If specified, load the specified EasyBuild module(s). Multiple modules may be specified in a colon-delimited list.
    --max_threads           If needed, limit the number of concurrent processes
    --output_csv            Name of the CSV produced by the analysis script (default: output.csv)
    --output_img            Name of the image produced by the analysis script (default: output.tif)

Webpage: https://github.com/FredHutch/quantify-view-cells
    """.stripIndent()
}


workflow {

    // Show help message if the user specifies the --help flag at runtime
    if (params.help || !params.function_call || !params.sample_sheet || !params.output){
        // Invoke the function above which prints the help message
        helpMessage()
        // Exit out and do not run anything else
        exit 0
    }

    // If there are assets specified
    if (params.assets) {

        // Set up a channel pointing to those files
        assets_ch = Channel
            .fromPath(params.assets.tokenize(","))
            .toSortedList()

    } else {

        // If no assets were specified
        // Make an empty channel
        assets_ch = Channel.empty()

    }

    // Parse the parameter string
    // This process has two outputs: 
    //      parameter_list (a CSV)
    //      function_calls (a list of files)
    // The `param_ix` column in the parameter_list
    // correponds to the filename in function_calls
    parse_parameters()

    // Add an index column to the manifest (image_ix)
    validate_sample_sheet(
        Channel.fromPath(params.sample_sheet)
    )

    // Make a channel which includes
        // The numeric index of each image
        // The image file
        // The numeric index of the parameter combination
        // The script to run
    img_ch = validate_sample_sheet
        .out
        .splitCsv(header: true)
        .flatten()
        .map {row -> [row.image_ix, file(row.uri)]}
        .combine(
            parse_parameters
                .out
                .function_calls
                .flatten()
                .map {i -> [i.name, i]}
        )

    // Run the script on each file
    run_script(
        img_ch,
        assets_ch
    )

    // Collect and compress the image outputs
    collect_images(
        run_script.out.img.toSortedList()
    )

    // Collect and summarize the tables
    collect_tables(
        run_script.out.csv.toSortedList()
    )

}

process parse_parameters {

    // Load the module(s) specified by the user
    module params.module

    // Publish the CSV of parameter combinations
    publishDir params.output, mode: "copy", overwrite: true, pattern: "parameter.combinations.csv"

    output:
        path "parameter.combinations.csv", emit: parameter_list
        path "function_calls/*", emit: function_calls

"""#!/bin/bash

parse_parameters.py \
    --parameters "${params.parameters}" \
    --czifile "${params.czifile}" \
    --function-call "${params.function_call}" \
    --output parameter.combinations.csv \
    --script-folder function_calls

"""

}

process collect_images {

    publishDir params.output, mode: "copy", overwrite: true
    input:
        file "*"

    output:
        file "output_images.tar.gz"

"""#!/bin/bash

set -e

# Combine all of the images
tar cvfh output_images.tar *${params.output_img}

# Compress the tar
gzip output_images.tar
"""

}

process collect_tables {

    // Load the module(s) specified by the user
    module params.module
    publishDir params.output, mode: "copy", overwrite: true
    input:
        file "*"

    output:
        file "feature.info.csv.gz"
        file "image.summary.csv.gz"

"""#!/usr/bin/env python3

import os
import pandas as pd

def parse_csv(fp):

    # Get the image_ix and params_ix from the filename
    image_ix, params_ix = fp.replace(".${params.output_csv}", "").split(".", 1)

    # Read the CSV
    df = pd.read_csv(fp)

    # Append the image_ix and params_ix
    df = df.assign(
        image_ix=image_ix,
        params_ix=params_ix,
    )

    # Return the data
    return df


# Combine all of the tables, appending a column indicating the image index and the parameter index
df = pd.concat(
    [
        parse_csv(fp)
        for fp in os.listdir(".")
        if fp.endswith(".${params.output_csv}")
    ]
)

# Write out the combined per-feature table
df.to_csv("feature.info.csv.gz", index=None)

# Function to test if all values in a Series are numeric
def is_numeric(v):
    return v.apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).notnull(
    ).all()

# Get the list of columns in the table which are all numeric
num_cols = [c for c, v in df.items() if is_numeric(v)]

# Summarize each metric per image
summary = df.groupby(["image_ix", "params_ix"]).apply(
    lambda d: d.reindex(columns=num_cols).median()
)

# Write out the summary table
summary.to_csv("image.summary.csv.gz", index=None)

"""

}

process run_script {
  
    // Load the module(s) specified by the user
    module params.module
    maxForks params.max_threads
  
    input:
        tuple val(image_ix), path("${params.czifile}"), val(params_ix), path("custom_function.m")
        path assets

    output:
        path "${image_ix}.${params_ix}.${params.output_csv}", emit: csv
        path "${image_ix}.${params_ix}.${params.output_img}", emit: img

    script:
"""#!/bin/bash

set -e

matlab -nodisplay -nosplash -nodesktop -r "custom_function ; exit"

# Add the index to the output filenames
mv "${params.output_csv}" "${image_ix}.${params_ix}.${params.output_csv}"
mv "${params.output_img}" "${image_ix}.${params_ix}.${params.output_img}"
"""

}

process validate_sample_sheet {
  
    input:
        path sample_sheet_csv

    output:
        file "${sample_sheet_csv}"

    script:
"""#!/bin/bash

set -e

# Make sure that the sample sheet has 'uri' in the first row
if [[ \$(head -1 "${sample_sheet_csv}" | grep -c uri) == 0 ]]; then

    # Report the error
    echo "Sample sheet must contain a column named 'uri'"

    # Prevent any further analysis
    rm ${sample_sheet_csv}

else

    # Add an index column to the sample sheet
    i=0
    cat "${sample_sheet_csv}" | \
    while read line; do
        if (( \$i == 0 )); then
            echo image_ix,\$line
        else
            echo \$i,\$line
        fi
        let "i=i+1"
    done \
    > TEMP && \
    mv TEMP "${sample_sheet_csv}"

fi
"""

}