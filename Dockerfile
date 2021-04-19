FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
MAINTAINER Maite Urra
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        pkg-config \
        rsync \
        unzip \
        python3.8 \
        python3-pip \
        vim \
        nano \
        && \
    apt-get clean

COPY requirements.txt /neural-question-generation/requirements.txt
WORKDIR /neural-question-generation
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./ /neural-question-generation

CMD /bin/bash
