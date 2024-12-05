## Evaluation

We evaluate our models on the CV-Bench, MMStar, RealWorldQA, and OK-VQA benchmarks.

```bash
# install evaluation specific dependencies
pip install -e .["eval"]
pip install -e lmms-eval/
```

### CV-Bench

```bash
# prepare benchmark
git lfs install
cd datasets/eval && git clone https://huggingface.co/datasets/nyu-visionx/CV-Bench & cd ../..

# run eval on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/cv_bench.sh shi-labs/OLA-VLM-CLIP-ViT-Llama3-8b ola_vlm_clip_llama3 llava_llama_3
```

### MMStar

```bash
# prepare benchmark
git lfs install
cd datasets/eval && git clone https://huggingface.co/datasets/Lin-Chen/MMStar & cd ../..

# run eval on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mmstar.sh shi-labs/OLA-VLM-CLIP-ViT-Llama3-8b ola_vlm_clip_llama3 llava_llama_3
```

### RelworldQA (RWQA) and OK-VQA

```bash
# run on 4 GPUs
accelerate launch --num_processes=4 -m lmms_eval --model llava --model_args pretrained=shi-labs/OLA-VLM-CLIP-ViT-Llama3-8b,conv_template=llava_llama_3,attn_implementation="eager",device_map="" --tasks realworldqa,ok_vqa --batch_size 1 --log_samples --log_samples_suffix ola_vlm_clip_llama3 --output_path datasets/eval/results/ola_vlm_clip_llama3
```
