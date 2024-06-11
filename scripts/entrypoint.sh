#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the virtual environment
source /opt/virtualenv/bin/activate

# Check if scan path is provided
scan_path="$1"
if [ -z "$scan_path" ]; then
    echo "Please provide a scan path, cannot find it or is empty."
    echo "Usage: infer_ensemble.sh <scan_path>"
    exit 1
fi

# Define the model path
model_path="/opt/mindGlide/models/_20240404_conjurer_trained_dice_7733.pt"

# Set the PYTHONPATH
export PYTHONPATH="/opt:/opt/monai:${PYTHONPATH}"

# Define the command to run
command="python /opt/mindGlide/mindGlide/run_inference.py --model_file_paths ${model_path} --scan_path ${scan_path}"

# Print and execute the command
echo $command
exec $command