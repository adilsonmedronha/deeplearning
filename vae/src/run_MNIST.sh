#!/bin/bash
RUN_NAME="MNIST_test"
RESULT_PATH="results/$RUN_NAME"

python run.py \
    --run_name $RUN_NAME \
    --dataset_path "../../../../datasets/MNIST" \
    --dataset_name MNIST \
    --img_shape 1 28 28 \
    --batch_size 128 \
    --epochs 5 \
    --z_dim 10 \
    --n_images 2 \
    --lr 0.0001 \
    --w_init_method xavier \
    --model_save_path_best_loss_train "$RESULT_PATH/weights/best_train/" \
    --model_save_path_best_loss_val "$RESULT_PATH/weights/best_val/" \
    --model_save_path_last "$RESULT_PATH/weights/" \
    --path2results "$RESULT_PATH/imgs/" \
    --path2gif "/gif/MNIST_$RUN_NAME.gif"