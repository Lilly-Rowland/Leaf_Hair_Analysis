FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /workdir

ARG HOST_UID
ARG HOST_GID
ARG HOST_USER

RUN addgroup --gid $HOST_GID $HOST_USER
RUN adduser --disabled-password --gecos '' --uid $HOST_UID --gid $HOST_GID $HOST_USER 

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install python3-setuptools
RUN apt-get -y install python3-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install python3-opencv --no-install-recommends 

RUN pip install --upgrade pip


COPY requirements.txt /workdir

RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt

RUN chown -R $HOST_USER:$HOST_USER /workdir

USER $HOST_USER

RUN pip install gradio

WORKDIR /workdir/Leaf_Hair_Analysis

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"