#!/bin/bash

# Check if the file with aliases is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <aliases_file>"
    exit 1
fi

# Read the file with aliases
aliases_file=$1
output_dir="fasta_files"
mkdir -p "$output_dir"

# Loop through each line in the file
while IFS= read -r line; do
    # Extract the alias and PDB ID
    alias=$(echo "$line" | awk '{print $1}')
    pdb_id=$(echo "$line" | awk '{print $2}')

    # Run the Python script to download and export the FASTA file
    python download_and_export_fasta.py "$pdb_id" "$output_dir"
done < "$aliases_file"

# Merge all FASTA files into one
merged_fasta="merged_sequences.fasta"
cat "$output_dir"/*.fasta > "$merged_fasta"

echo "Merged FASTA file created: $merged_fasta"
