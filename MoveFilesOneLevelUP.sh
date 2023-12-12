#!/bin/bash

# Assuming the root directory is 'dir'
root_dir=~/storage/Dataset/MI_Reservoir_Dataset/

# Function to move files and remove processed folders
move_files() {
    local src_dir="$1"
    local dest_dir="$2"

    # Move all files inside the 'processed' folder to its parent folder
    find "$src_dir" -type f -exec mv -t "$dest_dir" {} +

    # Delete the now-empty 'processed' folder
    rmdir "$src_dir"
}

# Find all the parent directories containing "_S_" in their name
find "$root_dir" -type d -name "*_DD" | while read parent_dir; do

    # Find the 'processed' folder inside the current parent directory
    processed_folder=$(find "$parent_dir" -type d -name "processed" -print -quit)

    echo "Handling processed folder: $processed_folder"

    # Check if the 'processed' folder exists
    if [ -n "$processed_folder" ]; then
        # Get the immediate parent directory of the 'processed' folder
        # and combine it with the parent directory containing "_S_" to construct the destination directory
        dest_dir=$(dirname "$processed_folder")
        dest_dir="$parent_dir/${dest_dir##*/}"

        # Move files and remove processed folder, preserving nested folder structure
        move_files "$processed_folder" "$dest_dir"
    fi
done