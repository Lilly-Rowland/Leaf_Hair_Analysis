FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /workdir

ARG HOST_UID
ARG HOST_GID
ARG HOST_USER

RUN addgroup --gid $HOST_GID $HOST_USER
RUN adduser --disabled-password --gecos '' --uid $HOST_UID --gid $HOST_GID $HOST_USER 

# Print the working directory and list its contents for debugging
RUN echo "Current working directory:" && pwd && echo "Listing files in the working directory:" && ls -la

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

# COPY ./deployments/leaf_hair_analysis/requirements.txt /workdir
RUN echo pwd
RUN echo hi
COPY requirements.txt /workdir

RUN pip install --no-cache-dir -r requirements.txt

RUN chown -R $HOST_USER:$HOST_USER /workdir

USER $HOST_USER

#ENTRYPOINT ["python", "./app.py"]