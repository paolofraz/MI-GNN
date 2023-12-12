#!/bin/bash

source activate PyMIGNN

# Define the Python file to execute
python_file="/home/frazzetp/MI-GNN/GraphGym/run/scripts/agg_runs_only.py"

# Define the text file with cfg arguments
arguments_file="/home/frazzetp/MI-GNN/SLURM/MIRES/baseline_NCI1.txt"

# Define the maximum number of parallel processes
max_processes=4

# Counter for tracking the number of active processes
active_processes=0

# Read each line from the arguments file
while IFS= read -r argument || [[ -n "$argument" ]]; do
    # Display the argument being executed
    echo "Executing Python file with argument: $argument"
    # Execute the Python file with the current argument
    python "$python_file" --cfg "$argument" --repeat 0 &

    # Increment the counter for active processes
    active_processes=$((active_processes + 1))

    # If the maximum number of parallel processes is reached, wait for any process to finish
    if [[ "$active_processes" -ge "$max_processes" ]]; then
        wait -n
        active_processes=$((active_processes - 1))
    fi
done < "$arguments_file"

wait
