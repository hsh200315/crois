FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow

COPY ./requirements.txt /mlflow/requirements.txt
COPY ./entrypoint.sh /mlflow/entrypoint.sh

RUN pip install -U pip &&\
    pip install -r /mlflow/requirements.txt

# MinIO Client 설치
RUN cd /tmp && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/bin/mc