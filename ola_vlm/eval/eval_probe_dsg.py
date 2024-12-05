import argparse
import torch

from ola_vlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ola_vlm.conversation import conv_templates
from ola_vlm.model.builder import load_pretrained_model
from ola_vlm.utils import disable_torch_init
from ola_vlm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from ola_vlm.model.aux_heads.oneformer_head import OneFormerHead
from ola_vlm.model.aux_heads.depth_anything_v2.dpt import DepthAnythingV2
from transformers import OneFormerProcessor

from diffusers import (
    DPMSolverMultistepScheduler,
    StableUnCLIPImg2ImgPipeline,
)

from PIL import Image
import json
import os
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import math

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

def get_gen_feats(pipe, image):
    with torch.no_grad():
        clip_ims = pipe.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        feat = pipe.image_encoder(clip_ims).image_embeds
    return feat

def get_dav2_feats(dav2, image):
    image = image.resize((336, 336))
    image = np.array(image)
    with torch.no_grad():
        feat = dav2.infer_image(image, is_dsg=True)
    return feat[-1][0]

def get_seg_feats(oneformer, oneformer_processor, image):
    img = image.resize((768, 768))
    inputs = oneformer_processor(img, ["panoptic"], return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
    with torch.no_grad():
        feats = oneformer.forward_features(**inputs)
    return feats


def predict(args):

    mode = args.mode

    name = args.model_path.split("/")[-1]
    os.makedirs(f"plots/probe_scores/{name}/", exist_ok=True)

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    if "llama3" in model_name.lower():
        conv_mode = "llava_llama_3"
    elif "phi" in model_name.lower():
        conv_mode = "llava_phi_3"

    images, prompts, answers = prepare_coco(args.json_file)

    images = get_chunk(images, args.num_chunks, args.chunk_idx)
    prompts = get_chunk(prompts, args.num_chunks, args.chunk_idx)
    answers = get_chunk(answers, args.num_chunks, args.chunk_idx)

    if mode == "gen":
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(f"stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

    elif mode == "seg":
        oneformer_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        oneformer = OneFormerHead.from_pretrained("shi-labs/oneformer_coco_swin_large")
        oneformer = oneformer.to("cuda")
    
    elif mode == "depth":
        dav2_cfg = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        dav2_backbone = DepthAnythingV2(**dav2_cfg)
        dav2_backbone.load_state_dict(torch.load("depth_anything_v2_vitl.pth", map_location='cpu'))
        dav2_backbone = dav2_backbone.to("cuda")
                

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
    

    os.makedirs(f"plots/probe_scores/{name}/{mode}/", exist_ok=True)
    
    if os.path.exists(f"plots/probe_scores/{name}/{mode}/{args.num_chunks}_{args.chunk_idx}.json"):
        with open(f"plots/probe_scores/{name}/{mode}/{args.num_chunks}_{args.chunk_idx}.json", 'r') as f:
            diff_dict = json.load(f)
    else:
        diff_dict = {}
    
    i = 0
    from tqdm import tqdm
    for fname, prompt, answer in tqdm(zip(images, prompts, answers), total=len(prompts)):
        
        conv = conv_templates[conv_mode].copy()
        image = load_image(fname)
        image = image.resize((640, 640))
    
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
                image_sizes=[image_size],
            )
        
        if mode == "gen":
            embeds = out.image_embs
            feats = get_gen_feats(pipe, image)
        elif mode == "depth":
            embeds = out.depth_embs
            embeds = [emb[0][0] for emb in embeds]
            feats = get_dav2_feats(dav2_backbone, image)
        elif mode == "seg":
            embeds = out.seg_embs
            feats = get_seg_feats(oneformer, oneformer_processor, image)

        layer_diff = {}
        for i, emb in enumerate(embeds):
            emb = emb.to("cuda")
            layer_diff[layers[i]] = 1.0 - torch.nn.CosineEmbeddingLoss(reduction="mean")(
                    emb.reshape(1, -1).float(), feats.reshape(1, -1).float(), 
                    torch.ones(len(emb)).to(feats.device)
                ).cpu().item()
        diff_dict[fname.split("/")[-1]] = layer_diff

        if i % 200 == 0:
            # Save progress intermittently
            with open(f"plots/probe_scores/{name}/{mode}/{args.num_chunks}_{args.chunk_idx}.json", 'w') as f:
                json.dump(diff_dict, f, indent=2)

        i += 1
    
    with open(f"plots/probe_scores/{name}/{mode}/{args.num_chunks}_{args.chunk_idx}.json", 'w') as f:
        json.dump(diff_dict, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="shi-labs/OLA-VLM-CLIP-ConvNeXT-Llama3-8b")
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
