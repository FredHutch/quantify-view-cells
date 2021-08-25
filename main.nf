#!/usr/bin/env nextflow

// Using DSL-2
nextflow.enable.dsl=2

// Set default parameters
params.help = false
params.script = false
params.assets = false
params.sample_sheet = false
params.output = false
params.concat_n = 100
params.module = false

// Function which prints help message text
def helpMessage() {
    log.info"""
Usage:

nextflow run FredHutch/quantify-view-cells <ARGUMENTS>

Required Arguments:
    --script                Script to be run on each image
    --assets                Any additional files which should be present in the working
                            directory in order for the script to run.
                            Multiple files (or folders) can be specified with a comma-delimited
                            list, while also supporting wildcards.
    --sample_sheet          List of images to process using the specified analysis
    --output                Path to output directory

Optional Arguments:
    --concat_n              Number of tabular results to combine/concatenate in the first round (default: 100)
    --module                If specified, load the specified EasyBuild module

Webpage: https://github.com/FredHutch/quantify-view-cells
    """.stripIndent()
}


workflow {

    // Show help message if the user specifies the --help flag at runtime
    if (params.help || !params.script || !params.sample_sheet || !params.output){
        // Invoke the function above which prints the help message
        helpMessage()
        // Exit out and do not run anything else
        exit 0
    }

    // Point to the script
    script = file(params.script)

    // If there are assets specified
    if (params.assets) {

        // Set up a channel pointing to those files
        assets_ch = Channel
            .fromPath(params.assets.split(","))
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

}

process validate_sample_sheet {
  
    input:
        path sample_sheet_csv

    output:
        file "${sample_sheet_csv}"

    script:
"""#!/bin/bash

# Add an index column to the sample sheet
cat "${sample_sheet_csv}" | \
awk -f validate_sample_sheet.awk > \
TEMP && \
mv TEMP "${sample_sheet_csv}"

"""

}