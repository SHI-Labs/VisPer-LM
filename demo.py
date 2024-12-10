import gradio as gr
import os
import torch
import numpy as np

from ola_vlm.constants import DEFAULT_IMAGE_TOKEN

from ola_vlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from ola_vlm.conversation import conv_templates, SeparatorStyle
from ola_vlm.model.builder import load_pretrained_model
from ola_vlm.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler
from transformers import OneFormerProcessor
from ola_vlm.model.aux_heads.oneformer_head import OneFormerHead
from ola_vlm.ola_utils import visualize_oneformer_masks_on_image, oneformer_prepare_panoptic_instance_prediction
import matplotlib
from PIL import Image, ImageDraw, ImageFont
import argparse
import math

from transformers import TextIteratorStreamer
from threading import Thread

def make_grid(pil_images, layer_indices=None):
    new_images = []
    new_captions = []
    
    # Resize images and prepare captions
    for i, pil_image in enumerate(pil_images):
        pil_image = pil_image.resize((256, 256))
        new_images.append(pil_image)
        if layer_indices is not None:
            new_captions.append(f"Layer: {layer_indices[i]}")
        else:
            new_captions.append(f"Layer: {i+1}")
    
    images = new_images
    captions = new_captions

    width, height = images[0].size
    font_size = 18

    # Calculate the number of rows and columns for the grid
    images_per_row = min(len(images), 4)  # Max 4 images per row
    row_count = math.ceil(len(images) / images_per_row)
    total_width = width * images_per_row
    total_height = height * row_count

    # Create a new blank image
    new_image = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(new_image)

    # Load a default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Place images and captions in the grid
    for i, (image, caption) in enumerate(zip(images, captions)):
        row = i // images_per_row
        col = i % images_per_row
        x_offset = col * width
        y_offset = row * height
        
        # Paste the image
        new_image.paste(image, (x_offset, y_offset))
        
        # Calculate text and background positions
        text_width, text_height = draw.textsize(caption, font=font)
        text_position = (x_offset + 10, y_offset + height - text_height - 10)
        background_position = (
            text_position[0] - 5,
            text_position[1] - 5,
            text_position[0] + text_width + 5,
            text_position[1] + text_height + 5,
        )

        # Draw background rectangle and text
        draw.rectangle(background_position, fill="white", outline="black")
        draw.text(text_position, caption, fill="black", font=font)
    
    return new_image

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
                state_dict[key] = f.get_tensor(key)

    # Load the state dict into the model
    model.load_state_dict(state_dict, strict=False)
    return model

# os.environ['GRADIO_TEMP_DIR'] = './gradio_tmp'
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

argparser = argparse.ArgumentParser()
argparser.add_argument("--server_name", default="0.0.0.0", type=str)
argparser.add_argument("--port", default="6324", type=str)
argparser.add_argument("--model-path", default="shi-labs/OLA-VLM-CLIP-ViT-Llama3-8b", type=str)
argparser.add_argument("--PT-model-path", default="shi-labs/pretrain_dsg_OLA-VLM-CLIP-ViT-Llama3-8b", type=str)
argparser.add_argument("--model-base", type=str, default=None)
argparser.add_argument("--num-gpus", type=int, default=1)
argparser.add_argument("--conv-mode", type=str, default="llava_llama_3")
argparser.add_argument("--temperature", type=float, default=0.2)
argparser.add_argument("--max-new-tokens", type=int, default=512)
argparser.add_argument("--num_frames", type=int, default=16)
argparser.add_argument("--load-8bit", action="store_true")
argparser.add_argument("--load-4bit", action="store_true")
argparser.add_argument("--debug", action="store_true")

args = argparser.parse_args()
model_path = args.model_path
conv_mode = args.conv_mode
filt_invalid="cut"
model_name = get_model_name_from_path(args.PT_model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
model = reload_from_ckpt(args.model_path, model)
our_chatbot = None

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(f"stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

oneformer_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
oneformer = OneFormerHead.from_pretrained("shi-labs/oneformer_coco_swin_large").to("cuda")

gen_layer_indices = model.config.image_gen["img_layer_indices"].split("-")
seg_layer_indices = model.config.image_seg["seg_layer_indices"].split("-")
depth_layer_indices = model.config.image_depth["depth_layer_indices"].split("-")


def clear_history():
    state =conv_templates[conv_mode].copy()
    return (state, state.to_gradio_chatbot(), "", None, None, None, None) + (disable_btn,) * 5

def add_text(state, imagebox, textbox, image_process_mode):
    if state is None:
        state = conv_templates[conv_mode].copy()

    if imagebox is not None:
        textbox = DEFAULT_IMAGE_TOKEN + '\n' + textbox
        image = Image.open(imagebox).convert('RGB')

    if imagebox is not None:
        textbox = (textbox, image, image_process_mode)

    state.append_message(state.roles[0], textbox)
    state.append_message(state.roles[1], None)

    yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)

def get_gen_images(out):
    img_embeds = out.image_embs
    if len(img_embeds) == 0:
        return None
    images = []
    for img_embed in img_embeds:
        gen_image = pipe(image_embeds=img_embed.squeeze(1),
                num_inference_steps=25,
            ).images[0]
        images.append(gen_image)
    grid_image = make_grid(images, gen_layer_indices)
    return grid_image

def get_depth_images(out, org_size):
    depth_preds = out.depth_preds

    if len(depth_preds) == 0:
        return None
    depths = []

    for i, depth_pred in enumerate(depth_preds):
        depth = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min()) * 255.0
        depth = depth.squeeze(0).cpu().numpy()
        depth = depth.astype(np.uint8)
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)       
        depth = Image.fromarray(depth)
        depth = depth.resize(org_size)
        depths.append(depth)
    grid_image = make_grid(depths, depth_layer_indices)
    return grid_image

def get_seg_images(out, image):
    seg_embs = out.seg_embs
    
    if len(seg_embs) == 0:
        return None
    
    seg_preds = []
    inputs = oneformer_processor(image, ["semantic"], return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(out.logits.device, out.logits.dtype)
    inputs["task_inputs"] = inputs["task_inputs"].to(out.logits.device, out.logits.dtype)
    backbone_features = oneformer.get_backbone_feats(**inputs)
    for i, seg_emb in enumerate(seg_embs):
        pred = oneformer.get_masks(**inputs, backbone_last_feature=seg_emb.float(), all_backbone_features=backbone_features)
        pred = oneformer_processor.post_process_panoptic_segmentation(
                                pred, target_sizes=[image.size[::-1]]
                            )[0]
        pred_msk, pred_cls = oneformer_prepare_panoptic_instance_prediction(**pred, oneformer=oneformer)
        pred = visualize_oneformer_masks_on_image(image, pred_msk, pred_cls)
        seg_preds.append(pred)
    grid_image = make_grid(seg_preds, seg_layer_indices)
    return grid_image

def delete_text(state, image_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)

def regenerate(state, image_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5

def get_interm_outs(state):
    prompt = state.get_prompt()
    images = state.get_images(return_pil=True)
    #prompt, image_args = process_image(prompt, images)

    if images is not None and len(images) > 0:
        if len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")
            
            #images = [load_image_from_base64(image) for image in images]
            image_sizes = [image.size for image in images]
            inp_images = process_images(images, image_processor, model.config)

            if type(inp_images) is list:
                inp_images = [image.to(model.device, dtype=torch.float16) for image in images]
            else:
                inp_images = inp_images.to(model.device, dtype=torch.float16)
        else:
            inp_images = None
            image_sizes = None
        image_args = {"images": inp_images, "image_sizes": image_sizes}
    else:
        inp_images = None
        image_args = {}

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    interm_outs = model.get_visual_interpretations(
                input_ids,
                **image_args
         )
    
    depth_outs = get_depth_images(interm_outs, image_sizes[0]) 
    seg_outs =  get_seg_images(interm_outs, images[0])
    gen_outs = get_gen_images(interm_outs)

    return depth_outs, seg_outs, gen_outs

# @spaces.GPU
def generate(state, temperature, top_p, max_output_tokens):
    prompt = state.get_prompt()
    images = state.get_images(return_pil=True)
    #prompt, image_args = process_image(prompt, images)

    ori_prompt = prompt
    num_image_tokens = 0

    if images is not None and len(images) > 0:
        if len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")
            
            #images = [load_image_from_base64(image) for image in images]
            image_sizes = [image.size for image in images]
            images = process_images(images, image_processor, model.config)

            if type(images) is list:
                images = [image.to(model.device, dtype=torch.float16) for image in images]
            else:
                images = images.to(model.device, dtype=torch.float16)
        else:
            images = None
            image_sizes = None
        image_args = {"images": images, "image_sizes": image_sizes}
    else:
        images = None
        image_args = {}

    max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
    max_new_tokens = max_output_tokens
    do_sample = True if temperature > 0.001 else False
    stop_str = state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

    max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

    if max_new_tokens < 1:
        return
    
    thread = Thread(target=model.generate, kwargs=dict(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        **image_args
    ))
    thread.start()
    generated_text = ''
    for new_text in streamer:
        generated_text += new_text
        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)]
        state.messages[-1][-1] = generated_text
        yield (state, state.to_gradio_chatbot(), "", None) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
    
    yield (state, state.to_gradio_chatbot(), "", None) + (enable_btn,) * 5
    
    torch.cuda.empty_cache()

txt = gr.Textbox(
    scale=4,
    show_label=False,
    placeholder="Enter text and press enter.",
    container=False,
)


title = "<h1 style='margin-bottom: -10px; text-align: center'>OLA-VLM: Elevating Visual Perception in Multimodal LLMs with Auxiliary Embedding Distillation</h1>"
description = "<p style='font-size: 16px; margin: 5px; font-weight: w300; text-align: center'> <a href='https://praeclarumjj3.github.io/' style='text-decoration:none' target='_blank'>Jitesh Jain</a> &nbsp;&nbsp <a href='https://zyang-ur.github.io/' style='text-decoration:none' target='_blank'>Zhengyuan Yang</a> &nbsp;&nbsp <a href='https://www.humphreyshi.com/home' style='text-decoration:none' target='_blank'>Humphrey Shi<sup>*</sup></a> &nbsp;&nbsp <a href='https://www.humphreyshi.com/home' style='text-decoration:none' target='_blank'>Jianfeng Gao<sup>*</sup></a> &nbsp;&nbsp <a href='https://jwyang.github.io/' style='text-decoration:none' target='_blank'>Jianwei Yang<sup>*</sup></a></p>" \
            + "<p style='font-size: 12px; margin: 5px; font-weight: w300; text-align: center'><sup>*</sup>Equal Advising</p>" \
            + "<p style='font-size: 16px; margin: 5px; font-weight: w600; text-align: center'> <a href='https://praeclarumjj3.github.io/ola_vlm/' target='_blank'>Project Page</a> | <a href='https://youtu.be/' target='_blank'>Video</a> | <a href='https://arxiv.org/abs/' target='_blank'>ArXiv</a> | <a href='https://github.com/SHI-Labs/OLA-VLM' target='_blank'>Github</a></p>"

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the [License](https://huggingface.co/lmsys/vicuna-7b-v1.5) of Vicuna-v1.5, [License](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE) of LLaVA, [Terms of Use](https://cocodataset.org/#termsofuse) of the COCO dataset, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""


textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
with gr.Blocks(title="OLA-VLM", theme=gr.themes.Default(), css=block_css) as demo:
    state = gr.State()

    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=4):
            imagebox = gr.Image(label="Input Image", type="filepath")
            image_process_mode = gr.Radio(
                ["Crop", "Resize", "Pad", "Default"],
                value="Default",
                label="Preprocess for non-square image", visible=False)

            # with gr.Accordion("Parameters", open=False) as parameter_row:
            with gr.Row():
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
            max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="OLA-VLM",
                height=300,
                layout="panel",
            )
            textbox.render()
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=False, visible=False)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=False, visible=False)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=False, visible=False)
                #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
                submit_btn = gr.Button(value="Send", variant="primary")

    with gr.Accordion("Representations from selected layers of the LLM (expects only a single image input)", open=False) as interm_out:
        inter_vis_btn = gr.Button(value="✨ Visualize")
        with gr.Row():
            depth_box = gr.Image(label="depth", type="pil", visible=True)
            seg_box = gr.Image(label="seg", type="pil", visible=True)
            gen_box = gr.Image(label="gen", type="pil", visible=True)
    
    gr.Examples(examples=[
            [f"assets/cars.jpg", "Which car is in front: the blue or the brown one?"],
            [f"assets/pb.jpg", "Where is the bulding located with respect to the man?"],
        ], inputs=[imagebox, textbox], cache_examples=False)

    # gr.Markdown(tos_markdown)
    # gr.Markdown(learn_more_markdown)
    # url_params = gr.JSON(visible=False)

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]

    inter_vis_btn.click(
        get_interm_outs,
        [state],
        [depth_box, seg_box, gen_box],
    )

    clear_btn.click(
        clear_history,
        None,
        [state, chatbot, textbox, imagebox, depth_box, gen_box, seg_box] + btn_list,
        queue=False
    )

    regenerate_btn.click(
        delete_text,
        [state, image_process_mode],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        generate,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot, textbox, imagebox] + btn_list,
    )
    textbox.submit(
        add_text,
        [state, imagebox, textbox, image_process_mode],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        generate,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot, textbox, imagebox] + btn_list,
    )

    submit_btn.click(
        add_text,
        [state, imagebox, textbox, image_process_mode],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        generate,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot, textbox, imagebox] + btn_list,
    )

demo.queue(
    status_update_rate=10,
    api_open=False
).launch(share=True)
demo.queue()