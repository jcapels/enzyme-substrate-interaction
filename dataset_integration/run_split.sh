podman build . -t dataset_enzymes_split
podman run -v /home/jcapela/enzyme-substrate-interaction/dataset_integration/docker_volume/:/app/:Z -d --name dataset_enzymes_split dataset_enzymes_split