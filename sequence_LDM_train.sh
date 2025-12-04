#!/bin/bash
sequence_LDM_config="configs/MRNet-sequence-LDM.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/training_config_ddp.yaml --main_process_port=8888 train_sequence_LDM.py --config=$sequence_LDM_config