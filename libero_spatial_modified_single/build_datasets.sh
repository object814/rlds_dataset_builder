#!/bin/bash

declare -A TASKS=(
    [0]='pick up the black bowl on the ramekin and place it on the plate'
    [1]='pick up the black bowl from table center and place it on the plate'
    [2]='pick up the black bowl next to the ramekin and place it on the plate'
    [3]='pick up the black bowl on the wooden cabinet and place it on the plate'
    [4]='pick up the black bowl next to the plate and place it on the plate'
    [5]='pick up the black bowl on the stove and place it on the plate'
    [6]='pick up the black bowl between the plate and the ramekin and place it on the plate'
    [7]='pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate'
)


for i in "${!TASKS[@]}"
do
    TASK_NAME="${TASKS[$i]}"
    DATASET_NAME="libero_spatial_modified_${i}"
    echo "Building dataset for $TASK_NAME with dataset name $DATASET_NAME"
    export TASK_NAME=$TASK_NAME
    export TASK_IDX=$i
    TFDS_DATA_DIR="/data/zhouhy/Datasets/modified_libero_rlds_single" tfds build --overwrite
done