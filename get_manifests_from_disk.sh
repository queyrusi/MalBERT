#!/bin/bash

######################################################################################
# Script Name: get_manifests.sh
# Description: Decompress APK files using the apktool tool.
#              It processes two input folders (good and malware) each containing 
#              folder that in turn contain an APK.
#              For each APK file, it checks if the corresponding decompressed 
#              directory already exists inside a [goodware|malware] temp folder.
#              If not, it uses apktool to decompress the APK file into the 
#              specified output directory. Then, it finds the AndroidManifest.xml,
#              renames it and puts it inside the data/manifests/D0/[goodware|malware] 
#              folder. Finally, it deletes the decompressed folder.
#       Usage: sh get_manifests.sh "/mnt/mydisk/dataset-D0/d0_benign_store" \
#                                "/mnt/mydisk/dataset-D0/d0_malicious_store" \
#                                "data/decompressed_apks/D0" \
#                                "data/manifests/D0"
#        Note: Script meant to be run on machine containing the disk with the APKs.
######################################################################################

# Set the source directories
goodware_dir="$1"
malware_dir="$2"

# Set the decompression destination directories
decompressed_dir="$3"

# Set the manifest destination directories
manifest_dir="$4"

# Start timer
start=$(date +%s)

# Count the number of folders inside goodware_dir
# goodware_folder_count=$(find "$goodware_dir" -type d | wc -l)
goodware_folder_count=5263
echo "Total count of folders inside goodware_dir: $goodware_folder_count"

# Count the number of folders inside malware_dir
# malware_folder_count=$(find "$malware_dir" -type d | wc -l)
malware_folder_count=5954
echo "Total count of folders inside malware_dir: $malware_folder_count"

current_folder_count=1
for apk in "$goodware_dir"/*/*.apk; do

    base=$(basename "$apk" .apk)

    # Check if the manifest file already exists
    if [ -f "$manifest_dir/goodware/${base}_manifest.xml" ]; then
        continue
    fi

    # Decompress the APK file
    if [ ! -d "$decompressed_dir/goodware/$base" ]; then
        apktool d "$apk" -o "$decompressed_dir/goodware/$base"
    fi

    # Find the .xml manifest inside the decompressed folder
    manifest=$(find "$decompressed_dir/goodware/$base" -maxdepth 1 -type f -name AndroidManifest.xml)

    # Copy the manifest to data/manifests/D0/goodware
    cp "$manifest" "$manifest_dir/goodware/${base}_manifest.xml"

    # Delete the decompression folder
    rm -rf "$decompressed_dir/goodware/$base"
done

current_folder_count=1
for apk in "$malware_dir"/*/*.apk; do
    
    base=$(basename "$apk" .apk)

    # Check if the manifest file already exists
    if [ -f "$manifest_dir/malware/${base}_manifest.xml" ]; then
        continue
    fi

    # Decompress the APK file
    if [ ! -d "$decompressed_dir0/malware/$base" ]; then
        apktool d "$apk" -o "$decompressed_dir/malware/$base"
    fi

    # Find the .xml manifest inside the decompressed folder. The manifest must be imediately at the root of the folder
    manifest=$(find "$decompressed_dir/malware/$base" -maxdepth 1 -type f -name AndroidManifest.xml)

    # Copy the manifest to data/manifests/D0/malware
    cp "$manifest" "$decompressed_dir/malware/${base}_manifest.xml"

    # Delete the decompression folder
    rm -rf "$decompressed_dir/malware/$base"
done

# End timer
end=$(date +%s)

# Calculate time taken
runtime=$((end-start))
echo "Time taken: $runtime seconds"