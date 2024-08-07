#!/bin/bash

"""
Example for running inferences using model-1 on repository06032024_DM_6-8-2024_3dpi_1 and 
saving the leaf masks.
"""

LOCAL_DIR=/local/workdir/$(whoami)
PROJECT_DIR=Leaf_Hair_Analysis
IMAGE_NAME=leaf_hair_analysis_image
GPUS=8
MEM="16G"

# Choose one of the following tasks: train, metrics, train_and_infer, ablation
TASK="inferr"

# Here, the variable that are unused are commented out, but they don't have to be

# MODEL_PATH= "path/to/model.pth" # infer, metrics*, train_and_infer
# LEAF_IMAGES="training_images" # train, metrics, train_and_infer, ablation
# ANNOTATIONS="annotations/labelbox_coco.json" # train, metrics, train_and_infer, ablation
# BATCH_SIZE=32 # train, metrics, train_and_infer
# EPOCHS=100 # train, metrics, train_and_infer
# ARCHITECTURE="DeepLabV3" # train, train_and_infer
# LOSS="dicebce" # train, metrics, train_and_infer
# BALANCE=True # train, metrics, train_and_infer
# GPU_INDEX=1 # train, metrics, train_and_infer
# SUBSET_SIZE=200 # metrics
# LEARNING_RATE=.001 # train, train_and_infer
LEAVES_TO_INFERENCE="repository06032024_DM_6-8-2024_3dpi_1" # infer, train_and_infer
# RESULTS_FILE="results.xlsx" # infer, train_and_infer <-- the default results file will be used instead
MAKE_HAIR_MASK=True # infer, train_and_infer
USE_MODEL_1=True # infer
# ABLATION_RESULTS_FILE="path/to/ablation/results/output/path" # ablation


# Required arguments should always be first
docker1 run --gpus $GPUS --rm\
	--shm-size=$MEM \
	docker.io/biohpc_$(whoami)/$IMAGE_NAME \
	python3 app.py $TASK \
	--image-dir $LEAVES_TO_INFERENCE \
	--make-hair-mask $MAKE_HAIR_MASK \
    --use-model-1 $USE_MODEL_1


"""
Example for training a model with cross entropy loss, 10 epochs, and the rest is default
"""

LOCAL_DIR=/local/workdir/$(whoami)
PROJECT_DIR=Leaf_Hair_Analysis
IMAGE_NAME=leaf_hair_analysis_image
GPUS=8
MEM="16G"

# Choose one of the following tasks: train, metrics, train_and_infer, ablation
TASK="train"

# Here, the unused variables are deleted
EPOCHS=10 # train, metrics, train_and_infer
LOSS="xe" # train, metrics, train_and_infer



docker1 run --gpus $GPUS --rm\
	--shm-size=$MEM \
	docker.io/biohpc_$(whoami)/$IMAGE_NAME \
	python3 app.py $TASK \
    --epochs 10 \
    --loss $LOSS \
	
	
"""
Example for running metrics on model-1 with all default settings
"""

LOCAL_DIR=/local/workdir/$(whoami)
PROJECT_DIR=Leaf_Hair_Analysis
IMAGE_NAME=leaf_hair_analysis_image
GPUS=8
MEM="16G"

# Choose one of the following tasks: train, metrics, train_and_infer, ablation
TASK="metrics"

# Check comments for what sub-arguments are required (*) and optional for the requested task
MODEL_PATH="models/model-1.pth" #"path/to/model.pth" # infer, metrics*, train_and_infer



docker1 run --gpus $GPUS --rm\
	--shm-size=$MEM \
	docker.io/biohpc_$(whoami)/$IMAGE_NAME \
	python3 app.py $TASK \
    --model-path $MODEL_PATH \
	
	