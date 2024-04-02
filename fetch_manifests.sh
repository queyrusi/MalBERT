#!/bin/bash

################################################################################
# Script Name: fetch_manifests.sh
# Description: This script processes directories and moves AndroidManifest.xml
#              files to a target directory.
################################################################################

# Directories to process
dirs=("goodware" "malware")

# Iterate over directories
for dir in "${dirs[@]}"; do
    # Create target directory if it doesn't exist
    mkdir -p "data/manifests/D0/$dir"
    # Iterate over folders in directory
    for folder in "data/decompressed_apks/D0/$dir"/*; do
        # Check if the folder contains an AndroidManifest.xml file
        if [ -f "$folder/AndroidManifest.xml" ]; then
            # Get the base name of the folder
            base=$(basename "$folder")
            # Rename and move the AndroidManifest.xml file
            mv "$folder/AndroidManifest.xml" "data/manifests/D0/$dir/${base}_manifest.xml"
        fi
    done
done