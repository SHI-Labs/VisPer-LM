#!/bin/bash

export WANDB_PROJECT= "OLA-VLM"
export WANDB_NAME="pretrain_dsg_OLA-VLM-CLIP-ViT-Llama3-8b"

# Base LLM choices: 
# Llama3-8b: meta-llama/Meta-Llama-3-8B-Instruct (llava_llama_3)
# Phi3-4k-mini: microsoft/Phi-3-mini-4k-instruct (llava_phi_3)

# Base encoder choices:
# CLIP-ViT-L: openai/clip-vit-large-patch14-336
# CLIP-ConvNeXT-XXL: laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup-res768

# 8 GPUs
deepspeed ola_vlm/train/ola_vlm_train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --version llava_llama_3 \
    --mode gen-depth-seg \
    --layer_indices d18-20_s10-18_g12-20 \
    --num_task_tokens 8 \
    --loss_weights d0.5_s0.5_g0.5 \
    --contrastive_loss_weight 0.3 \
    --image_generator stabilityai/stable-diffusion-2-1-unclip \
    --image_segmentor shi-labs/oneformer_coco_swin_large \
    --depth_estimator depth_anything_v2_vitl.pth \
    --data_path datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder datasets/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir outputs/pretrain_dsg_OLA-VLM-CLIP-ViT-Llama3-8b \
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
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb