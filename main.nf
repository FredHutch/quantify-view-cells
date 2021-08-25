#!/usr/bin/env nextflow

// Using DSL-2
nextflow.enable.dsl=2

// Set default parameters
params.help = false
params.function_call = false
params.assets = false
params.sample_sheet = false
params.output = false
params.concat_n = 100
params.module = ""

// Function which prints help message text
def helpMessage() {
    log.info"""
Usage:

nextflow run FredHutch/quantify-view-cells <ARGUMENTS>

Required Arguments:
    --function_call         Formatted function call which will analyze an image
                            (named 'input.czi') and write an output (named 'output.mat')
    --assets                Any additional files which should be present in the working
                            directory in order for the script to run.
                            Multiple files (or folders) can be specified with a comma-delimited
                            list, while also supporting wildcards. Double wildcards cross directory boundaries.
    --sample_sheet          List of images to process using the specified analysis
    --output                Path to output directory

Optional Arguments:
    --concat_n              Number of tabular results to combine/concatenate in the first round (default: 100)
    --module                If specified, load the specified EasyBuild module(s). Multiple modules may be specified in a colon-delimited list.

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

    // Add an index column to the manifest
    validate_sample_sheet(
        Channel.fromPath(params.sample_sheet)
    )

    // Make a channel which includes
        // The numeric index of each image
        // The image file
        // Any additional assets
    img_ch = validate_sample_sheet
        .out
        .splitCsv(header: true)
        .flatten()
        .map {row -> [row.ix, file(row.uri)]}

    // Run the script on each file
    run_script(
        img_ch,
        assets_ch
    )

}

process run_script {
  
    // Load the module(s) specified by the user
    module params.module
  
    input:
        tuple val(ix), path("input.czi")
        path assets

    output:
        tuple val(ix), path("output.mat")

    script:
"""#!/bin/bash

set -e

matlab -nodisplay -nosplash -nodesktop -r "${params.function_call}"
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
            echo ix,\$line
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