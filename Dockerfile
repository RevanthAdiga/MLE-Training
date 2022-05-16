FROM python:3.8-slim-buster

RUN apt-get update && \
    apt-get install -y gcc make apt-transport-https ca-certificates build-essential

RUN python3 --version
RUN pip3 --version

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python setup.py install
RUN python ingest_data.py
RUN python train.py
RUN python score.py
RUN pytest tests/
