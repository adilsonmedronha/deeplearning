#!/bin/bash

RUN_NAME="CelebA_linear_teste_onnx"
RESULT_PATH="results/$RUN_NAME"

python run.py \
    --model_type "lvae"  \
    --run_name $RUN_NAME \
    --device "cuda"  \
    --dataset_path "/media/SSD/DATASETS/CelebA_tiny/CelebA/" \
    --dataset_name CelebA \
    --img_shape 3 64 64 \
    --batch_size 16 \
    --epochs 100 \
    --z_dim 20 \
    --n_images 2 \
    --lr 0.0001 \
    --w_init_method xavier \
    --model_save_path_best_loss_train "$RESULT_PATH/weights/best_train/" \
    --model_save_path_best_loss_val "$RESULT_PATH/weights/best_val/" \
    --model_save_path_last "$RESULT_PATH/weights/" \
    --path2results "$RESULT_PATH" \
    --path2gif "/gif/CelebA_$RUN_NAME.gif"