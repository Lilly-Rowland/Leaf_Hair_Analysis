#!/bin/bash

LOCAL_DIR=/local/workdir/$(whoami)
PROJECT_DIR=Leaf_Hair_Analysis
IMAGE_NAME=leaf_hair_analysis_image
GPUS=8
MEM="16G"

docker1 run --gpus $GPUS --rm\
	--shm-size=$MEM \
    -p 7860:7860 \
	docker.io/biohpc_$(whoami)/$IMAGE_NAME \
	bash -c "python3 -u gradio_app.py"