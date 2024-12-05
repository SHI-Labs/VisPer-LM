# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from ola_vlm.model.multimodal_projector.resampler import Resampler, TaskTokenResampler


class GenHead(nn.Module):

    def __init__(
        self,
        proj_config: dict = None,
        llm_hidden_size: int = 4096,
    ) -> None:
        super().__init__()

        self.projector = Resampler(
                dim=proj_config["output_dim"],
                depth=proj_config["depth"],
                dim_head=proj_config["dim_head"],
                heads=proj_config["num_heads"],
                num_queries=proj_config["num_tokens"],
                embedding_dim=llm_hidden_size,
                output_dim=proj_config["output_dim"],
                ff_mult=proj_config["ff_mult"],
            )
    
    def forward(
        self,
        llm_feats: torch.Tensor,
    ):
        gen_feats = self.projector(llm_feats)
        return gen_feats

class TaskTokenGenHead(nn.Module):

    def __init__(
        self,
        proj_config: dict = None,
        llm_hidden_size: int = 4096,
    ) -> None:
        super().__init__()

        self.projector = TaskTokenResampler(
                dim=proj_config["output_dim"],
                depth=proj_config["depth"],
                dim_head=proj_config["dim_head"],
                heads=proj_config["num_heads"],
                num_queries=proj_config["num_tokens"],
                embedding_dim=llm_hidden_size,
                output_dim=proj_config["output_dim"],
                ff_mult=proj_config["ff_mult"],
            )
    
    def forward(
        self,
        llm_feats: torch.Tensor,
        latents: torch.Tensor
    ):
        gen_feats = self.projector(llm_feats, latents)
        return gen_feats