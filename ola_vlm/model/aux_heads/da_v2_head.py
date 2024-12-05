import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ola_vlm.model.multimodal_projector.resampler import Resampler, TaskTokenResampler


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


class DAv2_Head(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DAv2_Head, self).__init__()
        
        self.embd_dims = {
            'vits': 1024,
            'vitb': 1024,
            'vitl': 1024, 
            'vitg': 1024,
        }
        
        self.depth_head = DPTHead(self.embd_dims[encoder], features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, features):
        patch_h, patch_w = 336 // 14, 336 // 14
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_feats(self, feats, image_size=(336, 336)):
        h, w = image_size
        depth = self.forward(feats)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.cpu().numpy()

def build_mlp(in_hidden_size, hidden_size):
    modules = [nn.Linear(in_hidden_size, hidden_size)]
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*modules)

def build_expand_mlp(in_hidden_size, hidden_size, out_size):
    modules = [nn.Linear(in_hidden_size, hidden_size)]
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, hidden_size))
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, out_size))
    return nn.Sequential(*modules)

class DepthProbeHead(nn.Module):
    def __init__(
        self, 
        llm_hidden_size=4096,
        proj_config=None,
    ):
        super(DepthProbeHead, self).__init__()
        
        self.linear_1 = build_mlp(llm_hidden_size, proj_config["output_dim"])
        self.linear_2 = build_mlp(llm_hidden_size, proj_config["output_dim"])
        self.linear_3 = build_mlp(llm_hidden_size, proj_config["output_dim"])
        self.linear_4 = build_mlp(llm_hidden_size, proj_config["output_dim"])

    #     self._init_weights()

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, llm_feats):

        features = [(self.linear_1(llm_feats), None),
            (self.linear_1(llm_feats), None),
            (self.linear_2(llm_feats), None),
            (self.linear_3(llm_feats), None)
        ]

        return features

class DepthHead(nn.Module):
    def __init__(
        self, 
        llm_hidden_size=4096,
        proj_config=None,
        use_intermediate_depth=False,
    ):
        super(DepthHead, self).__init__()
        
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

        self.use_intermediate_depth = use_intermediate_depth

        if self.use_intermediate_depth:            
            self.linear_1 = build_mlp(proj_config["output_dim"], proj_config["output_dim"])
            self.linear_2 = build_mlp(proj_config["output_dim"], proj_config["output_dim"])
            self.linear_3 = build_mlp(proj_config["output_dim"], proj_config["output_dim"])

    def forward(self, llm_feats):
        visual_feats = self.projector(llm_feats)

        features = []
        
        if self.use_intermediate_depth:    
            features.append((self.linear_1(visual_feats), None))
            features.append((self.linear_2(visual_feats), None))
            features.append((self.linear_3(visual_feats), None))
        
        features.append((visual_feats, None))

        return features

class TaskTokenDepthHead(nn.Module):
    def __init__(
        self, 
        proj_config=None,
        llm_hidden_size=4096,
        use_intermediate_depth=False,
    ):
        super(TaskTokenDepthHead, self).__init__()

        self.projector = TaskTokenResampler(
                dim=llm_hidden_size,
                depth=proj_config["depth"],
                dim_head=proj_config["dim_head"],
                heads=proj_config["num_heads"],
                num_queries=proj_config["num_tokens"],
                embedding_dim=llm_hidden_size,
                output_dim=proj_config["output_dim"],
                ff_mult=proj_config["ff_mult"],
            )
        self.use_intermediate_depth = use_intermediate_depth

        if self.use_intermediate_depth:
            self.linear_1 = build_mlp(proj_config["output_dim"], proj_config["output_dim"])
            self.linear_2 = build_mlp(proj_config["output_dim"], proj_config["output_dim"])
            self.linear_3 = build_mlp(proj_config["output_dim"], proj_config["output_dim"])

    def forward(self, llm_feats, latents):

        visual_feats = self.projector(llm_feats, latents)

        features = []
        
        if self.use_intermediate_depth:    
            features.append((self.linear_1(visual_feats), None))
            features.append((self.linear_2(visual_feats), None))
            features.append((self.linear_3(visual_feats), None))
        
        features.append((visual_feats, None))

        return features