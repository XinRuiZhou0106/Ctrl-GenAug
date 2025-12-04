#!/bin/bash
LDM_config="configs/compress_ratio_8_sd_config/MosMed-LDM.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/training_config_ddp.yaml --main_process_port=8888 train_LDM.py --config=$LDM_config