#!/bin/sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ola_vlm.eval.eval_probe_task \
        --model-path $1 --json-file datasets/coco/annotations/captions_val2017.json \
        --mode $2 --num-chunks $CHUNKS --chunk-idx $IDX &
done

wait

python -m ola_vlm.eval.get_probe_task_scores --ckpt $1 --mode $2