#!/bin/bash

######################################################################################
# Script Name: get_manifests.sh
# Description: Finds all APKs in a directory and extracts their manifests.
#       Usage: sh get_manifests.sh  --from-disk "/mnt/mydisk/dataset-D0/d0_benign_store" \
#                                "data/manifests/D0/goodware"
#        Note: Use --from-subfolders flag to specify APKs are in subdirectories
######################################################################################

# Set the source directories
source_dir="$1"

# Set the manifest destination directories
manifest_dir="$2"

# Start timer
start=$(date +%s)   

if [ "$1" = "--from-subfolders" ]; then
    apk_dir="$source_dir"/*/*.apk
else
    apk_dir="$source_dir"/*.apk
fi

for apk in $apk_dir; do

    base=$(basename "$apk" .apk)

    # Check if the manifest file already exists
    if [ -f "$manifest_dir/${base}_manifest.xml" ]; then
        continue
    fi

    # Launch axmldec to convert the binary form of the manifest to XML
    axmldec -o "$manifest_dir/${base}_manifest.xml" $apk

done

# End timer
end=$(date +%s)

# Calculate time taken
runtime=$((end-start))
echo "Time taken: $runtime seconds"