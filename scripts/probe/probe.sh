#!/bin/bash

export WANDB_PROJECT= "OLA-VLM"
export WANDB_NAME="probe_depth_ola-vlm-pt-ift"

# 8 GPUs
deepspeed ola_vlm/train/probe_dsg_train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --mode $1 \
    --model_name_or_path shi-labs/OLA-VLM-CLIP-ConvNeXT-Llama3-8b \
    --image_generator stabilityai/stable-diffusion-2-1-unclip \
    --image_segmentor shi-labs/oneformer_coco_swin_large \
    --depth_estimator depth_anything_v2_vitl.pth \
    --version llava_llama_3 \
    --data_path /mnt/vlpdatasets/sherlock/coco/annotations/captions_train2017.json \
    --image_folder datasets/coco \
    --vision_tower laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup-res768 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --tf32 True \
    --output_dir outputs/probe_${1}_ola-vlm-pt-ift \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb