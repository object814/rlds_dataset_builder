#!/bin/bash

declare -A TASKS=(
    [0]='pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo'
    [1]='pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo'
    [2]='pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo'
    [3]='pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo'
    [4]='pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo'
    [5]='pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo'
    [6]='pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo'
    [7]='pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo'
    [8]='pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo'
    [9]='pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo'
)


for i in "${!TASKS[@]}"
do
    TASK_NAME="${TASKS[$i]}"
    DATASET_NAME="libero_spatial_${i}"
    echo "Building dataset for $TASK_NAME with dataset name $DATASET_NAME"
    export TASK_NAME=$TASK_NAME
    export TASK_IDX=$i
    TFDS_DATA_DIR="/data/zhouhy/Datasets/LIBERO_spatial_rlds" tfds build --overwrite
done