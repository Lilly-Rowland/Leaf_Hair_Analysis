#!/bin/bash

# Get the current user's UID and GID
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(whoami)  # Replace with the desired username

# Build the Docker image with build arguments
docker1 build -t {{cookiecutter.image_name}} --no-cache \
  --build-arg HOST_UID=$HOST_UID \
  --build-arg HOST_GID=$HOST_GID \
  --build-arg HOST_USER=$HOST_USER \
  -f Dockerfile /workdir/$HOST_USER
