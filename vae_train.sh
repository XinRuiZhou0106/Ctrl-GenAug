#!/bin/bash
vae_config="configs/compress_ratio_8_vae_config/MosMed-vae.yaml"
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --config_file configs/training_config_ddp.yaml --main_process_port=8888 train_vae.py --config=$vae_config