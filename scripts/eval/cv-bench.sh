#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ola_vlm.eval.model_cvbench_loader \
        --model-path $1 --path datasets/eval/CV-Bench \
        --answers-file datasets/eval/results/$2/cv-bench/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS --chunk-idx $IDX --temperature 0 --conv-mode $3 &
done

wait

output_file=datasets/eval/results/$2/cv-bench/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat datasets/eval/results/$2/cv-bench/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ola_vlm/eval/eval_cv_bench.py --results_file $output_file

