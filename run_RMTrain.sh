#!/bin/bash

accelerate launch \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"/local_ssd_data/pangxl/tulu_test/llama-3.1-tulu-3-8b-preference-mixture": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"/local_ssd_data/pangxl/tulu_test/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path /home/pangxl/open-instruct/output/sft_8b \
    --chat_template tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir output/rm_8b \
    --gradient_checkpointing \
    --with_tracking