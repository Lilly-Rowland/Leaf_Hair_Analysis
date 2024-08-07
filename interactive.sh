#!/bin/bash

LOCAL_DIR=/local/workdir/$(whoami)
PROJECT_DIR=leaf_hair_analysis
IMAGE_NAME=leaf_hair_analysis_image
GPUS=8
MEM="16G"

docker1 run --gpus $GPUS -it\
	--shm-size=$MEM \
	-e DATA_DIR=/workdir/deployments/data \
	-e RESULTS_DIR=/workdir/deployments/$PROJECT_DIR/results \
	docker.io/biohpc_$(whoami)/$IMAGE_NAME \
	bash
