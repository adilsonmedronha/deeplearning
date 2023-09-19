#!/bin/bash

RUN_NAME="CelebA_linear_new"
RESULT_PATH="results/$RUN_NAME"

python run.py \
    --run_name $RUN_NAME \
    --model_type "lvae"  \
    --dataset_path "/media/SSD/DATASETS/CelebA_tiny/CelebA/" \
    --dataset_name CelebA \
    --img_shape 3 128 128 \
    --batch_size 32 \
    --epochs 10 \
    --z_dim 20 \
    --n_images 2 \
    --lr 0.00001 \
    --w_init_method xavier \
    --model_save_path_best_loss_train "$RESULT_PATH/weights/best_train/" \
    --model_save_path_best_loss_val "$RESULT_PATH/weights/best_val/" \
    --model_save_path_last "$RESULT_PATH/weights/" \
    --path2results "$RESULT_PATH/imgs/" \
    --path2gif "/gif/CelebA_$RUN_NAME.gif"