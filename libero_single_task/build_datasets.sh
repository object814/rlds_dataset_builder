#!/bin/bash

# libero_object_no_noops
declare -A TASKS=(
    [0]='pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo'
    [1]='pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo'
    [2]='pick_up_the_salad_dressing_and_place_it_in_the_basket_demo'
    [3]='pick_up_the_ketchup_and_place_it_in_the_basket_demo'
    [4]='pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo'
    [5]='pick_up_the_milk_and_place_it_in_the_basket_demo'
    [6]='pick_up_the_cream_cheese_and_place_it_in_the_basket_demo'
    [7]='pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo'
    [8]='pick_up_the_orange_juice_and_place_it_in_the_basket_demo'
    [9]='pick_up_the_butter_and_place_it_in_the_basket_demo'
)

for i in "${!TASKS[@]}"
do
    TASK_NAME="${TASKS[$i]}"
    DATASET_NAME="libero_object_no_noops"
    echo "Building dataset for $TASK_NAME with dataset name $DATASET_NAME_$i"
    export TASK_NAME=$TASK_NAME
    export TASK_IDX=$i
    export DATASET_NAME=$DATASET_NAME
    TFDS_DATA_DIR="/data/zhouhy/Datasets/LIBERO_RLDS/datasets/libero_object_no_noops" tfds build --overwrite
done