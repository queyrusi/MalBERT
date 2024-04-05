#!/bin/bash

######################################################################################
# Script Name: get_manifests.sh
# Description: Finds all APKs in a directory and extracts their manifests.
#       Usage: sh get_manifests.sh "/mnt/mydisk/dataset-D0/d0_benign_store" \
#                                "data/manifests/D0/goodware"
#        Note: Use --from-subfolders flag to specify APKs are in subdirectories
######################################################################################


# Start timer
start=$(date +%s)   

if [ "$1" = "--from-subfolders" ]; then

    # Set the source directories
    source_dir="$2"

    # Set the manifest destination directories
    manifest_dir="$3"

    apk_dir="$source_dir"/*/*.apk
else

    # Set the source directories
    source_dir="$1"

    # Set the manifest destination directories
    manifest_dir="$2"

    apk_dir="$source_dir"/*.apk
fi

for apk in $apk_dir; do

    base=$(basename "$apk" .apk)

    # Check if the manifest file already exists
    if [ -f "$manifest_dir/${base}_manifest.xml" ]; then
        echo "Manifest for $base already exists. Skipping..."
        continue
    fi

    # Unzip the APK and store AndroidManifest.xml in a variable
    base=$(basename "$apk" .apk)
    unzip -p "$apk" "AndroidManifest.xml" > "/tmp/${base}.xml"

    # Launch axmldec to convert the binary form of the manifest to XML
    axmldec -o "$manifest_dir/${base}_manifest.xml" "/tmp/${base}.xml"

    # Remove the temporary file
    rm "/tmp/${base}.xml"

done

# End timer
end=$(date +%s)

# Calculate time taken
runtime=$((end-start))
echo "Time taken: $runtime seconds"