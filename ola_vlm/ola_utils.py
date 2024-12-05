from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np


import torch.distributed as dist
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import diffdist.functional as diff_dist

from typing import List, Optional
from torchvision.ops import masks_to_boxes
import io


def visualize_oneformer_masks_on_image(
    image: torch.Tensor,
    masks: List[torch.Tensor],
    classes: List[str],
    save_path: Optional[str] = None,
):
    """
    inputs:
        image: torch.Tensor of shape (3, H, W)
        masks: List[torch.Tensor] of len NUM_MASKS
        classes: List[str] of len NUM_MASKS
        save_path: Optional[str] path to save the visualization
    returns:
        pil_image: PIL.Image with masks overlayed on the image
    """

    def _show_mask(mask, class_name, ax, random_color=False):
        mask = mask.cpu()
        box = masks_to_boxes(mask.unsqueeze(0))[0]
        x0, y0, x1, y1 = box
        x = (x0 + x1) / 2
        y = (y0 + y1) / 2
        if random_color:
            color = np.concatenate(
                [np.random.random(3), np.array([0.6])], axis=0
            )
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        ax.text(x, y, class_name, fontsize="x-small")

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(np.array(image))  # Convert to HWC format for plt
    ax.set_autoscale_on(False)
    for mask, class_name in zip(masks, classes):
        _show_mask(mask, class_name, ax=ax, random_color=True)
    plt.axis("off")
    plt.tight_layout()

    # Save figure to a BytesIO object and convert to PIL.Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    pil_image = Image.open(buf)

    # Optionally save the PIL image
    if save_path is not None:
        pil_image.save(save_path)

    plt.close(fig)
    return pil_image

def oneformer_prepare_panoptic_instance_prediction(
    segmentation: torch.Tensor, segments_info: dict, oneformer
):
    masks = []
    classes = []

    for segment in segments_info:
        id = segment["id"]
        label_id = segment["label_id"]
        label = oneformer.config.id2label[label_id]
        mask = segmentation == id
        masks.append(mask.float())
        classes.append(label)

    return masks, classes

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()

def calculate_contrastive_loss(preds, targets, logit_scale):
    batch_size = preds.shape[0]
    if is_dist_avail_and_initialized():
        labels = torch.arange(batch_size, dtype=torch.long, device=preds.device) + batch_size * dist.get_rank()
    else:
        labels = torch.arange(batch_size, dtype=torch.long, device=preds.device)

    preds = F.normalize(preds.flatten(1), dim=-1)
    targets = F.normalize(targets.flatten(1), dim=-1)

    if is_dist_avail_and_initialized():
        logits_per_img = preds @ dist_collect(targets).t()
    else:
        logits_per_img = preds @ targets.t()

    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    loss_contrastive = F.cross_entropy(logits_per_img * logit_scale, labels, reduction="none")
    return loss_contrastive

def silog_loss(depth_est, depth_gt, variance_focus=0.5):
    mask = (depth_gt > 0).detach()
    if mask.sum() == 0:
        return torch.tensor(0.0).to(depth_est)
    d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
    loss = torch.sqrt(torch.pow(d, 2).mean() -
                        variance_focus * torch.pow(d.mean(), 2)) * 1.0
    return loss

def make_grid(images, pil_images):
    # Assuming each image is the same size
    
    new_images = []
    new_captions = []
    for image, pil_image in zip(images, pil_images):
        new_images.append(image)
        pil_image = pil_image.resize((image.size[0], image.size[1]))
        new_images.append(pil_image)
        new_captions.append("Predicted")
        new_captions.append("GT")
    
    images = new_images
    captions = new_captions

    width, height = images[0].size
    font_size = 14
    caption_height = font_size + 10

    # Calculate the size of the final image
    images_per_row = min(len(images), 16)  # Round up for odd number of images
    row_count = (len(images) + 1) // images_per_row
    total_width = width * images_per_row
    total_height = (height + caption_height) * row_count

    # Create a new blank image
    new_image = Image.new("RGB", (total_width, total_height), "white")

    draw = ImageDraw.Draw(new_image)

    for i, (image, caption) in enumerate(zip(images, captions)):
        row = i // images_per_row
        col = i % images_per_row
        x_offset = col * width
        y_offset = row * (height + caption_height)
        
        new_image.paste(image, (x_offset, y_offset))
        text_position = (x_offset + 10, y_offset + height)
        draw.text(text_position, caption, fill="red", font_size=font_size)
    
    return new_image

def visualize_masks(anns, rgb_image):
    if len(anns) == 0:
        return rgb_image
    
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img_array = np.array(rgb_image)
    masked_image = np.ones(img_array.shape)
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)
        
        masked_image[m] = (color_mask * 255).astype(np.uint8)
    
    img_array = img_array * 0.35 + masked_image * 0.65
    
    img_array = img_array.astype(np.uint8)
    ax.imshow(img_array)
    overlayed_img = Image.fromarray(img_array)
    
    return overlayed_img