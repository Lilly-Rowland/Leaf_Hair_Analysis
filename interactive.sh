#!/bin/bash

LOCAL_DIR=/local/workdir/$(whoami)
PROJECT_DIR=Leaf_Hair_Analysis
IMAGE_NAME=leaf_hair_analysis_image
GPUS=8
MEM="16G"

docker1 run --gpus $GPUS -it\
	--shm-size=$MEM \
	docker.io/biohpc_$(whoami)/$IMAGE_NAME \
	bash
