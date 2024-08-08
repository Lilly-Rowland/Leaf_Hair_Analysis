#!/bin/bash

LOCAL_DIR=/local/workdir/$(whoami)
PROJECT_DIR=Leaf_Hair_Analysis
IMAGE_NAME=leaf_hair_analysis_image
GPUS=8
MEM="16G"

# Choose one of the following tasks: train, metrics, train_and_infer, ablation
TASK="infer"

# Check comments for what sub-arguments are required (*) and optional for the requested task

MODEL_PATH="models/model-2.pth" # "path/to/model.pth" # infer, metrics*, train_and_infer
LEAF_IMAGES="training_images/dataset_2_training_images" # train, metrics, train_and_infer, ablation
ANNOTATIONS="annotations/labelbox_coco.json" # train, metrics, train_and_infer, ablation
BATCH_SIZE=32 # train, metrics, train_and_infer
EPOCHS=100 # train, metrics, train_and_infer
ARCHITECTURE="DeepLabV3" # train, train_and_infer
LOSS="dice" # train, metrics, train_and_infer
BALANCE=True # train, metrics, train_and_infer
GPU_INDEX=1 # train, metrics, train_and_infer
SUBSET_SIZE=200 # metrics
LEARNING_RATE=.001 # train, train_and_infer
LEAVES_TO_INFERENCE="repository06032024_DM_6-8-2024_3dpi_1" # infer, train_and_infer
RESULTS_FOLDER="results_temp" # infer, train_and_infer
MAKE_HAIR_MASK=False # infer, train_and_infer
USE_MODEL_1=False # infer
ABLATION_RESULTS_FILE="path/to/ablation/results/output/path" # ablation

docker1 run --gpus $GPUS --rm\
	--shm-size=$MEM \
	docker.io/biohpc_$(whoami)/$IMAGE_NAME \
	python3 app.py $TASK \
	--image-dir $LEAVES_TO_INFERENCE \
	--model-path "models/model-2.pth" \
	--results-folder "repository06032024_DM_6-8-2024_3dpi_1_inferences_4_15_30" \
	--make-hair-mask $MAKE_HAIR_MASK
    # add the sub-arguments here. Check README or use --help tag for more details


# # Run in background:
# docker1 run --gpus $GPUS --rm \
#     --shm-size=$MEM \
#     docker.io/biohpc_$(whoami)/$IMAGE_NAME \
#     bash -c "nohup python3 -u app.py $TASK \
#     --image_dir $LEAVES_TO_INFERENCE \
#     --model-path 'models/model-2.pth' \
#     --results-folder 'repository06032024_DM_6-8-2024_3dpi_1_inferences_4_15_30' \
#     --make-hair-mask $MAKE_HAIR_MASK > inferences.log 2>&1 &"