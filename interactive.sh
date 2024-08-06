#!/bin/bash

LOCAL_DIR=/local/workdir/$(whoami)
PROJECT_DIR={{cookiecutter.project_name}}
API_KEY={{cookiecutter.wandb_key}}
IMAGE_NAME={{cookiecutter.image_name}}
GPUS={{cookiecutter.gpus}}
MEM={{cookiecutter.mem}}

docker1 run --gpus $GPUS -it\
	--shm-size=$MEM \
	-e DATA_DIR=/workdir/deployments/data \
	-e RESULTS_DIR=/workdir/deployments/$PROJECT_DIR/results \
	docker.io/biohpc_$(whoami)/$IMAGE_NAME \
	bash
