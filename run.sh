

podman build . -t cdhit_swiss_prot
podman run -v $(pwd)/pipeline/data/:/blast/data/:Z -d --rm --name cdhit_swiss_prot cdhit_swiss_prot 