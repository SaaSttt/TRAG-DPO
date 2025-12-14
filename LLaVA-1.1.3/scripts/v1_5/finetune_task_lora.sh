#!/bin/bash

export TRANSFORMERS_OFFLINE=1

include=localhost:0,1,2,3,4,5
model_name_or_path=TRAG-DPO/llava-v1.5-7b
data_path=TRAG-DPO/Qwen-VL/dataset/VQA/tomato_llava.json
image_folder=TRAG-DPO/Qwen-VL/dataset/images
vision_tower=TRAG-DPO/clip-vit-large-patch14-336

DEEPSPEED_ARGS="$DEEPSPEED_ARGS --no_local_rank"
export MODEL_TYPE=llava

# 移除了不支持的参数：--pin_memory, --persistent_workers, --use_flash_attention_2
deepspeed --include $include llava/train/train_mem.py \
    --lora_enable True --lora_r 8 --lora_alpha 16 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $model_name_or_path \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower TRAG-DPO/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora_tomato \
    --num_train_epochs 5 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to none