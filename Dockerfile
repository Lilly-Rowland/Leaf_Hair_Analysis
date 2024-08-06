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

COPY ./deployments/{{cookiecutter.project_name}}/requirements.txt /workdir

RUN pip install --no-cache-dir -r requirements.txt

RUN chown -R $HOST_USER:$HOST_USER /workdir

USER $HOST_USER

ENTRYPOINT ["python", "./app.py"]

# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# # Copy the requirements file into the container
# COPY requirements.txt /app/

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code to the container
# COPY . /app

# # Set the working directory in the container
# WORKDIR /app

# # Set the default command to run the script
# ENTRYPOINT ["python", "./app.py"]
