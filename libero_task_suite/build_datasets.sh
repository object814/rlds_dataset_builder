#!/bin/bash

# Set dataset name
DATASET_NAME="libero_90_no_noops"
DATA_DIR="/data/zhouhy/Datasets/LIBERO_RLDS/datasets/libero_90_no_noops"

echo "Building dataset for $DATASET_NAME"

# Export the dataset name so the builder can access it
export DATASET_NAME=$DATASET_NAME

TFDS_DATA_DIR=$DATA_DIR tfds build --overwrite