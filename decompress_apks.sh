#!/bin/bash

################################################################################
# Script Name: decompress_apks.sh
# Description: Decompress APK files using the apktool tool.
#              It processes two directories, "goodware" and "malware", and 
#              iterates over the APK files within each directory. For each 
#              APK file, it checks if the corresponding decompressed directory 
#              already exists. If not, it uses apktool to decompress the APK 
#              file into the specified output directory.
################################################################################



# Directories to process
dirs=("goodware" "malware")

# Iterate over directories
for dir in "${dirs[@]}"; do
    # Iterate over APK files in directory
    for apk in "data/apks/D0/$dir"/*.apk; do
        # Get the base name of the APK file without extension
        base=$(basename "$apk" .apk)
        # Check if the decompressed directory already exists
        if [ ! -d "data/decompressed_apks/D0/$dir/$base" ]; then
            # Decompress the APK file with apktool
            apktool d "$apk" -o "data/decompressed_apks/D0/$dir/$base"
        fi
    done
done