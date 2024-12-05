#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from ola_vlm.model import *
from ola_vlm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower() or 'clip' in model_name.lower() or 'sherlock' in model_name.lower() or 'dino' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from ola_vlm.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            if "phi" in model_name.lower():
                model = LlavaPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            elif "qwen" in model_name.lower():
                model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            else:    
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if "probe" in model_name.lower():
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if "phi" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    model = ProbeDSGLlavaPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                elif "qwen" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_base)
                    model = ProbeDSGLlavaQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    model = ProbeDSGLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            elif 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            elif "phi" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if "dsg" in model_name.lower():
                    model = OlaLlavaPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                elif "multi_enc" in model_name.lower():
                    model = MultiEncLlavaPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    model = LlavaPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            elif "qwen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if "dsg" in model_name.lower():
                    model = OlaLlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if "dsg" in model_name.lower():
                    model = OlaLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                elif "multi_enc" in model_name.lower():
                    model = MultiEncLlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'probe' in model_name.lower():
                if 'phi' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = ProbeDSGLlavaPhi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                elif 'qwen' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = ProbeDSGLlavaQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = ProbeDSGLlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif "phi" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if "dsg" in model_name.lower():
                    model = OlaLlavaPhi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                elif "multi_enc" in model_name.lower():
                    model = MultiEncLlavaPhi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaPhi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif "qwen" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if "dsg" in model_name.lower():
                    model = OlaLlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if "dsg" in model_name.lower():
                    model = OlaLlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
                elif "multi_enc" in model_name.lower():
                    model = MultiEncLlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower() or 'sherlock' in model_name.lower() or 'probe' in model_name.lower() or 'clip' in model_name.lower() or 'dino' in model_name.lower():
        
        if "convnext" in model_name.lower():
            model = reload_from_ckpt(model_path, model)

        vision_tower = model.get_vision_tower()
        
        if "multi_enc" in model_name.lower():
            model.get_model().init_encoders(model.config)
        
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)

        try:
            if vision_tower.device != model.device:
                vision_tower = vision_tower.to(model.device)
        except:
            pass
        
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 4096
    return tokenizer, model, image_processor, context_len


def reload_from_ckpt(model_path, model, cache_dir=None):
    import os
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download, list_repo_files

    state_dict = {}

    # Check if the path is a local directory or HF Hub model
    if os.path.isdir(model_path):
        # Local directory: Load safetensors files
        safetensors_paths = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.safetensors')]
    else:
        # HF Hub: Get list of safetensors files and download them
        repo_files = list_repo_files(model_path)
        safetensors_paths = [
            hf_hub_download(model_path, file_name, cache_dir=cache_dir)
            for file_name in repo_files if file_name.endswith('.safetensors')
        ]

    # Load safetensors files into the state_dict
    for path in safetensors_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "vision_tower" in key:
                    state_dict[key] = f.get_tensor(key)

    # Load the state dict into the model
    model.load_state_dict(state_dict, strict=False)
    return model