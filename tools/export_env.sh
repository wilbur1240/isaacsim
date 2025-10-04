#!/usr/bin/env bash

# Define the output file path
OUTPUT_FILE="/home/arg/isaac/docs/isaac_environment.yml"

# Check if the file already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "File $OUTPUT_FILE already exists."
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo    # Move to a new line
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Export cancelled."
        exit 0
    fi
    
    echo "Overwriting existing file..."
fi

# Export the conda environment
echo "Exporting conda environment to $OUTPUT_FILE..."
conda env export > "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Environment successfully exported to $OUTPUT_FILE"
else
    echo "Error: Failed to export environment"
    exit 1
fi