FROM ncbi/blast:latest

RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y emboss
RUN apt-get install -y pip

RUN pip install graph-part

WORKDIR /app

CMD graphpart needle --fasta-file unique_enzymes.fasta --threshold 0.6 --out-file graphpart_assignments_0_8.csv --test-ratio 0.1 --val-ratio 0.05 --threads 50