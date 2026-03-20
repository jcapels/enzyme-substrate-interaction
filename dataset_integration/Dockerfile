FROM ncbi/blast:latest

RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y emboss
RUN apt-get install -y pip

RUN pip install graph-part

WORKDIR /app

CMD sh run.sh