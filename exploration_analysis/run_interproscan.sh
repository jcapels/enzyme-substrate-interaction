podman run -d \
    -v $PWD/my_interproscan/interproscan-5.73-104.0/data:/opt/interproscan/data:Z \
    -v $PWD:/work:Z \
    --name interproscan_enzymes \
    interpro/interproscan:5.73-104.0 \
    --input /work/unique_enzymes_curated.fasta \
    --output-dir /work \
    --cpu 8 \
    -f tsv
    