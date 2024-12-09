# Integrate Predictive Embedding Optimization into your Custom MLLM

We provide a base class for our OLA-VLM, making it easy to integrate it into your own MLLM. Below, we outline the steps:

- Extend your MLLM class with the [`BaseOLA_VLM`](../ola_vlm/model/language_model/base_ola_vlm.py) class.

```python
from ola_vlm.language_model.base_ola_vlm import BaseOLA_VLM

class YourModelName(BaseOLA_VLM):
    ...
```

- Add the embedding losses to the forward pass:

```python
depth_preds, depth_embs, depth_loss, depth_l1_loss, depth_cont_loss = self.depth_emb_forward(pil_images, layer_states, depth_mask)
seg_embs, seg_loss, seg_l1_loss, seg_contrastive_loss = self.seg_emb_forward(pil_images, hidden_states, layer_states, seg_mask)
img_embs, gen_loss, gen_mse_loss, gen_con_loss = self.gen_emb_forward(pil_images, hidden_states, layer_states, gen_mask)

emb_loss = depth_loss + seg_loss + gen_loss

total_loss = text_ntp_loss + emb_loss
```

- You also need to make a few changes to your dataloader as shown in [`ola_vlm_train.py`](https://github.com/SHI-Labs/OLA-VLM/blob/e5133dc149a0bfec3646c3ce6cf79bb902ca187a/ola_vlm/train/ola_vlm_train.py#L774) for LLaVA-like models.

```python
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        ...

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ...

        if 'image' in sources[0]:
            ...
            pil_image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            ...
        else:
            ...
       ...

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['pil_image'] = pil_image
            data_dict['seg_mask'] = 1
            data_dict['depth_mask'] = 1
            data_dict['gen_mask'] = 1
        elif self.data_args.is_multimodal:
            try:
                crop_size = self.data_args.image_processor.crop_size
            except:
                crop_size = self.data_args.image_processor.size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['pil_image'] = Image.new('RGB', (crop_size['width'], crop_size['height']), color='black')
            data_dict['seg_mask'] = 0
            data_dict['depth_mask'] = 0
            data_dict['gen_mask'] = 0
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ...
        
        if 'pil_image' in instances[0]:
            pil_images = [instance['pil_image'] for instance in instances]
            batch['pil_images'] = pil_images

            seg_mask = [instance['seg_mask'] for instance in instances]
            batch['seg_mask'] = torch.tensor(seg_mask)
            
            depth_mask = [instance['depth_mask'] for instance in instances]
            batch['depth_mask'] = torch.tensor(depth_mask)
            
            gen_mask = [instance['gen_mask'] for instance in instances]
            batch['gen_mask'] = torch.tensor(gen_mask)
        
        return batch
```

- Lastly remember to add [these](https://github.com/SHI-Labs/OLA-VLM/blob/e5133dc149a0bfec3646c3ce6cf79bb902ca187a/ola_vlm/train/ola_vlm_train.py#L1149-L1266) lines before your training function.

Now, you are all set to optimize the embedding losses along with the next-token prediction during the MLLM training!