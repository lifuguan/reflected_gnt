import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm



class NeRFSemSegFPNHead(nn.Module):

    def __init__(self, args, feature_strides=[2,4,8,16], feature_channels=[128,128,128,128]):
        super(NeRFSemSegFPNHead, self).__init__()
        num_classes = args.num_classes
        conv_dims = 128
        self.scale_heads = nn.ModuleList()
        self.common_stride = 2
        # for stride, channels in zip(feature_strides, feature_channels):
        #     head_ops = []
        #     head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
        #     for k in range(head_length):
        #         norm_module = get_norm('GN', conv_dims)
        #         conv = Conv2d(
        #             channels if k == 0 else conv_dims,
        #             conv_dims,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1,
        #             bias=not 'GN',
        #             norm=norm_module,
        #             activation=F.relu,
        #         )
        #         weight_init.c2_msra_fill(conv)
        #         head_ops.append(conv)
        #         if stride != self.common_stride:
        #             head_ops.append(
        #                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        #             )
        #     self.scale_heads.append(nn.Sequential(*head_ops))
        
        # self.predictor = Conv2d(conv_dims, num_classes + 1, kernel_size=1, stride=1, padding=0)
        ### construct deeplabv3 decoder ###
        deeplabv3_dim = 512
        self.deeplabv3_decoder = DeepLabHead(deeplabv3_dim, num_classes+1, atrous_rates=(12,24,36))

    def original_index_to_downsampled_index(self, original_index):
        original_width, downsample_factor = 320, 2
        original_row = (original_index-1) // original_width
        original_col = (original_index-1) % original_width
        
        downsampled_row = original_row // downsample_factor
        downsampled_col = original_col // downsample_factor
        
        downsampled_index = downsampled_row * (original_width // downsample_factor) + downsampled_col
        
        return downsampled_index
    
    def forward(self, deep_feats, out_feats, select_inds):
        #######   replace feature map           #######
        _, D, H, W = deep_feats.shape
        if select_inds is not None:
            deep_feats = deep_feats.reshape(1, deep_feats.shape[1], -1).squeeze(0).permute(1,0)

            re_select_inds = []
            for select_ind in select_inds:
                re_select_inds.append(self.original_index_to_downsampled_index(select_ind))
            
            # distill loss
            device = deep_feats.device
            novel_feats = deep_feats[re_select_inds].detach()
            # novel_feats = deep_feats[re_select_inds]
            loss_distillation = F.cosine_embedding_loss(novel_feats, out_feats, torch.ones((len(re_select_inds))).to(device), reduction='mean')
        else:
            deep_feats = deep_feats.reshape(1, deep_feats.shape[1], -1).squeeze(0).permute(1,0)

        ####### constrcut feature pyramids and Decoder  #######
        # chunks = torch.chunk(deep_feats, 4, dim=1)
        # for i, chunk in enumerate(chunks):
        #     chunk = chunk.reshape(120, 160, chunk.shape[-1]).permute(2,0,1).unsqueeze(0) # 1, 512, h, w
        #     if i == 0:
        #         x = self.scale_heads[i](chunk)
        #     else:
        #         chunk = F.interpolate(chunk, scale_factor = 1/(2**i), mode='bilinear', align_corners=True, recompute_scale_factor=True)
        #         x = x + self.scale_heads[i](chunk)

        # out = self.predictor(x) # [1, cls_num, h, w]
        # out = F.interpolate(out, scale_factor = 2, mode='bilinear', align_corners=True)
        
        #######   construct deeplabv3 decoder  ########
        new_deep_feats = deep_feats.reshape(1, H, W, D).permute(0,3,1,2)
        out = self.deeplabv3_decoder(new_deep_feats)
        out = F.interpolate(out, scale_factor = 2, mode='bilinear', align_corners=True)
        if select_inds is not None:
            return out, loss_distillation
        else:
            return out

from typing import Sequence, Dict, Optional
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)
        
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)