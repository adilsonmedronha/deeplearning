#!/bin/bash

RUN_NAME="CelebA_pcvae"
RESULT_PATH="results/$RUN_NAME"

python run.py \
    --run_name $RUN_NAME \
    --model_type "pcvae"  \
    --dataset_path "/media/SSD/DATASETS/CelebA_tiny/CelebA/" \
    --dataset_name CelebA \
    --img_shape 3 125 125 \
    --batch_size 128 \
    --epochs 2 \
    --z_dim 20 \
    --n_images 2 \
    --lr 0.0001 \
    --w_init_method he \
    --model_save_path_best_loss_train "$RESULT_PATH/weights/best_train/" \
    --model_save_path_best_loss_val "$RESULT_PATH/weights/best_val/" \
    --model_save_path_last "$RESULT_PATH/weights/" \
    --path2results "$RESULT_PATH/imgs/" \
    --path2gif "/gif/CelebA_$RUN_NAME.gif"