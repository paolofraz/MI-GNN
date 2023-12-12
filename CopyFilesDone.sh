#!/bin/bash

# set the source directory
src_dir=/home/frazzetp/MI-GNN/GraphGym/run/results/baseline_graph_grid_weightdecay
# set the destination directory
dest_dir=/home/frazzetp/MI-GNN/GraphGym/run/results/weightdecay_partial

# loop through all the folders in the source directory
for folder in $(find "$src_dir" -type d); do
    # count the number of subfolders for the current folder
    subfolder_count=$(find "$folder" -mindepth 1 -maxdepth 1 -type d | wc -l)

    # if the folder has at least 3 subfolders, copy it to the destination directory
    if [ "$subfolder_count" -ge 3 ]; then
        cp -r "$folder" "$dest_dir"
        echo "$folder"
    fi
done
