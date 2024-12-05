import argparse
import torch

from ola_vlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ola_vlm.conversation import conv_templates
from ola_vlm.model.builder import load_pretrained_model
from ola_vlm.utils import disable_torch_init
from ola_vlm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from ola_vlm.model.aux_heads.oneformer_head import OneFormerHead
from transformers import OneFormerProcessor

from PIL import Image
import json
import os
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import math
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler

def prepare_da2k(path, is_eval=False):
    images = []
    prompts = []
    dir_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if "object" in root or is_eval:
                path = os.path.join(root, file)
                images.append(path)
                dir_name = path.split("/")[-2]
                prompt = "Describe the image."
                prompts.append(prompt)
                dir_names.append(dir_name)
    
    return images, prompts, dir_names


def parse_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

def prepare_coco(json_file):
    coco_data = parse_json(json_file)

    id_to_filename = {image["id"]: image["file_name"] for image in coco_data["images"]}
    images = []
    prompts = []
    answers = []

    im_ids = []

    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        if image_id in im_ids:
            continue
        file_name = id_to_filename[image_id]
        images.append(os.path.join(json_file.split("/annotations")[0], "val2017", file_name))
        im_ids.append(image_id)
        answers.append(caption)
        prompts.append("Describe the image in two lines.")

    return images, prompts, answers


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image

import glob

def list_image_files(directory):
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, extension)))
    return image_files

def prep_seginw(dir):
    image_files = list_image_files(dir)
    prompts = []
    for image_file in image_files:
        prompts.append("Describe the image")
    return image_files, prompts, prompts

def predict(args):

    mode = args.mode

    name = args.model_path.split("/")[-1]
    os.makedirs(f"plots/probes_task/{name}/", exist_ok=True)

    # Model
    disable_torch_init()

    if mode == 'gen' or mode == 'seg':
        images, prompts, answers = prepare_coco(args.json_file)
    elif mode == 'depth':
        images, prompts, answers = prepare_da2k("datasets/eval/DA-2K/images", is_eval=True)        
    
    images = get_chunk(images, args.num_chunks, args.chunk_idx)
    prompts = get_chunk(prompts, args.num_chunks, args.chunk_idx)
    answers = get_chunk(answers, args.num_chunks, args.chunk_idx)
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if mode == "gen":
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(f"stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

    elif mode == "seg":
        oneformer_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        oneformer = OneFormerHead.from_pretrained("shi-labs/oneformer_coco_swin_large")
        oneformer = oneformer.to("cuda")
                
    if "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "llama3" in model_name.lower():
        conv_mode = "llava_llama_3"
    elif "qwen" in model_name.lower():
        conv_mode = "qwen_1_5"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "phi" in model_name.lower():
        conv_mode = "llava_phi_3"

    set_seed(42)

    if mode == "gen":
        try:
            layers = model.config.image_gen["layer_indices"]
        except:
            layers = [i+1 for i in range(32)]
    elif mode == "depth":
        try:
            layers = model.config.image_depth["layer_indices"]
        except:
            layers = [i+1 for i in range(32)]
    elif mode == "seg":
        try:
            layers = model.config.image_seg["layer_indices"]
        except:
            layers = [i+1 for i in range(32)]

    from tqdm import tqdm
    for fname, prompt, answer in tqdm(zip(images, prompts, answers), total=len(prompts)):
        
        conv = conv_templates[conv_mode].copy()
        im = fname.split("/")[-1].split(".")[0]
        
        image = load_image(fname)
    
        image_size = image.size
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        inp = prompt
        if image is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        with torch.inference_mode():
            out = model.get_visual_interpretations(
                input_ids,
                images=image_tensor,
                image_sizes=image_size,
            )
        
        if mode == "seg":
            seg_embs = out.seg_embs
            inputs = oneformer_processor(image, ["semantic"], return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(out.logits.device, out.logits.dtype)
            inputs["task_inputs"] = inputs["task_inputs"].to(out.logits.device, out.logits.dtype)
            backbone_features = oneformer.get_backbone_feats(**inputs)
            for i, seg_emb in enumerate(seg_embs):
                pred = oneformer.get_masks(**inputs, backbone_last_feature=seg_emb.float(), all_backbone_features=backbone_features)
                pred = oneformer_processor.post_process_semantic_segmentation(
                                        pred, target_sizes=[image.size[::-1]]
                                    )[0]
                pred = pred.squeeze().cpu().numpy().astype(np.uint8)
                pred = Image.fromarray(pred)
                if not os.path.exists(f"plots/probes_task/{name}/seg/layer_{layers[i]}"):
                    os.makedirs(f"plots/probes_task/{name}/seg/layer_{layers[i]}", exist_ok=True)
                save_path = os.path.join(f"plots/probes_task/{name}/seg/layer_{layers[i]}", fname.split("/")[-1].replace("jpg", "png"))
                pred.save(save_path)
                
        
        elif mode == "gen":
            img_embeds = out.image_embs
            images = []

            for img_emb in img_embeds:
                gen_image = pipe(image_embeds=img_emb.squeeze(1),
                            num_inference_steps=25,
                        ).images[0]
                images.append(gen_image)
            
            for i, image in enumerate(images):
                image = image.resize((256, 256), Image.LANCZOS)
                if not os.path.exists(f"plots/probes_task/{name}/gen/layer_{layers[i]}"):
                    os.makedirs(f"plots/probes_task/{name}/gen/layer_{layers[i]}", exist_ok=True)
                save_path = os.path.join(f"plots/probes_task/{name}/gen/layer_{layers[i]}", fname.split("/")[-1])
                image.save(save_path)

        elif mode == "depth":
            depth_preds = out.depth_preds

            for i, depth_pred in enumerate(depth_preds):
                if not os.path.exists(f"plots/probes_task/{name}/depth/layer_{layers[i]}"):
                    os.makedirs(f"plots/probes_task/{name}/depth/layer_{layers[i]}", exist_ok=True)
                depth = depth_pred.squeeze(0).cpu().numpy() * 255.0
                depth = depth.astype(np.uint8)
                depth = Image.fromarray(depth)
                save_path = os.path.join(f"plots/probes_task/{name}/depth/layer_{layers[i]}", fname.split("/")[-1])
                depth.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="shi-labs/probe_depth_llava-1.5-pt-ift")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--json-file", type=str, default="datasets/coco/annotations/captions_val2017.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--mode", type=str, default="gen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()
    predict(args)
