#!/bin/bash

export WANDB_PROJECT= "VisPer-LM"
export WANDB_NAME="VisPer-LM-CLIP-ViT-Llama3-8b"

# Base LLM choices: 
# Llama3-8b: meta-llama/Meta-Llama-3-8B-Instruct (llava_llama_3)
# Phi3-4k-mini: microsoft/Phi-3-mini-4k-instruct (llava_phi_3)

# Base encoder choices:
# CLIP-ViT-L: openai/clip-vit-large-patch14-336
# CLIP-ConvNeXT-XXL: laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup-res768

# 8 GPUs
deepspeed ola_vlm/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path outputs/pretrain_dsg_VisPer-LM-CLIP-ViT-Llama3-8b \
    --version llava_llama_3 \
    --data_path datasets/llava_v1_5_mix665k.json \
    --image_folder datasets/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir outputs/VisPer-LM-CLIP-ViT-Llama3-8b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
