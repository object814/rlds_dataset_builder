#!/bin/bash


# The Script for building the RLDS datasets for LIBERO single tasks
# Run the script under this directory by: bash build_datasets.sh
# RLDS dataset for each individual task inside a task suite will be generated


# # libero_object_no_noops
# DATASET_NAME="libero_object_no_noops"
# DATA_DIR="/data/zhouhy/Datasets/LIBERO_RLDS/datasets/libero_object_no_noops"
# declare -A TASKS=(
#     [0]='pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo'
#     [1]='pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo'
#     [2]='pick_up_the_salad_dressing_and_place_it_in_the_basket_demo'
#     [3]='pick_up_the_ketchup_and_place_it_in_the_basket_demo'
#     [4]='pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo'
#     [5]='pick_up_the_milk_and_place_it_in_the_basket_demo'
#     [6]='pick_up_the_cream_cheese_and_place_it_in_the_basket_demo'
#     [7]='pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo'
#     [8]='pick_up_the_orange_juice_and_place_it_in_the_basket_demo'
#     [9]='pick_up_the_butter_and_place_it_in_the_basket_demo'
# )

# # libero_spatial_no_noops
# DATASET_NAME="libero_spatial_no_noops"
# DATA_DIR="/data/zhouhy/Datasets/LIBERO_RLDS/datasets/libero_spatial_no_noops"
# declare -A TASKS=(
#     [0]='pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo'
#     [1]='pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo'
#     [2]='pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo'
#     [3]='pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo'
#     [4]='pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo'
#     [5]='pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo'
#     [6]='pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo'
#     [7]='pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo'
#     [8]='pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo'
#     [9]='pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo'
# )

# # libero_goal_no_noops
# DATASET_NAME="libero_goal_no_noops"
# DATA_DIR="/data/zhouhy/Datasets/LIBERO_RLDS/datasets/libero_goal_no_noops"
# declare -A TASKS=(
#     [0]='put_the_wine_bottle_on_top_of_the_cabinet_demo'
#     [1]='open_the_top_drawer_and_put_the_bowl_inside_demo'
#     [2]='put_the_cream_cheese_in_the_bowl_demo'
#     [3]='open_the_middle_drawer_of_the_cabinet_demo'
#     [4]='put_the_bowl_on_the_plate_demo'
#     [5]='put_the_bowl_on_the_stove_demo'
#     [6]='push_the_plate_to_the_front_of_the_stove_demo'
#     [7]='put_the_wine_bottle_on_the_rack_demo'
#     [8]='turn_on_the_stove_demo'
#     [9]='put_the_bowl_on_top_of_the_cabinet_demo'
# )

# # libero_10_no_noops
# DATASET_NAME="libero_10_no_noops"
# DATA_DIR="/data/zhouhy/Datasets/LIBERO_RLDS/datasets/libero_10_no_noops"
# declare -A TASKS=(
#     [0]='KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo'
#     [1]='LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo'
#     [2]='LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo'
#     [3]='STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo'
#     [4]='KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo'
#     [5]='KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo'
#     [6]='LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo'
#     [7]='LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo'
#     [8]='LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo'
#     [9]='KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo'
# )

for i in "${!TASKS[@]}"
do
    TASK_NAME="${TASKS[$i]}"
    echo "Building dataset for $TASK_NAME with dataset name $DATASET_NAME_$i"
    export TASK_NAME=$TASK_NAME
    export TASK_IDX=$i
    export DATASET_NAME=$DATASET_NAME
    TFDS_DATA_DIR=$DATA_DIR tfds build --overwrite
done