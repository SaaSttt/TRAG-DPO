CUDA_DEVICES='0,1,2,3,4,5'
CUDA='0,1,2,3,4,5'

cd ./train/dpo || exit
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES deepspeed --include localhost:$CUDA ./train_dpo_2stages.py \
    --model_name_or_path TRAG-DPO/llava-v1.5-7b-all \
    --deepspeed ./scripts/zero3.json \
    --version v1 \
    --lora_enable True --lora_r 8 --lora_alpha 16 --mm_projector_lr 2e-5 \
    --data_path TRAG-DPO/dpo_data_notnull.json \
    --image_folder /data/datasets/zhuhaoran/image \
    --vision_tower TRAG-DPO/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora_dpo \
    --num_train_epochs 3 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to none \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True