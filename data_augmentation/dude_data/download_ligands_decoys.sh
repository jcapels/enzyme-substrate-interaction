#!/bin/bash

# Check if the file with targets is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <targets_file>"
    exit 1
fi

# Read the file with targets
targets_file=$1

# Loop through each target in the file
while IFS= read -r target; do
    # Create a directory for the target
    mkdir -p "$target"

    # Construct the URL for ligands
    ligands_url="https://dudez.docking.org/DUDE-Z-benchmark-grids/${target}/ligands.smi"

    # Use wget to download the ligands file into the target directory
    wget "$ligands_url" -O "${target}/ligands.smi"

    # Construct the URL for decoys
    decoys_url="https://dudez.docking.org/DUDE-Z-benchmark-grids/${target}/decoys.smi"

    # Use wget to download the decoys file into the target directory
    wget "$decoys_url" -O "${target}/decoys.smi"
done < "$targets_file"

echo "Downloads complete."
