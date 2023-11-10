import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm



class NeRFSemSegFPNHead(nn.Module):

    def __init__(self, args, feature_strides=[2,4,8,16], feature_channels=[128,128,128,128]):
        super(NeRFSemSegFPNHead, self).__init__()
        num_classes = args.num_classes
        self.downsample_factor = 2
        self.original_width = args.original_width
        self.original_height = args.original_height

        ### construct deeplabv3 decoder ###
        deeplabv3_dim = 512
        self.deeplabv3_decoder = DeepLabHead(deeplabv3_dim, num_classes+1, atrous_rates=(12,24,36))

    def original_index_to_downsampled_index(self, original_index):
        original_row = (original_index-1) // self.original_width
        original_col = (original_index-1) % self.original_width
        
        downsampled_row = original_row // self.downsample_factor
        downsampled_col = original_col // self.downsample_factor
        
        downsampled_index = downsampled_row * (self.original_width // self.downsample_factor) + downsampled_col
        
        return downsampled_index
    
    def forward(self, deep_feats, out_feats, select_inds):
        _, D, H, W = deep_feats.shape
        #######   replace feature map           #######
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

        new_deep_feats = deep_feats.reshape(1, H, W, D).permute(0,3,1,2)
        out = self.deeplabv3_decoder(new_deep_feats)
        out = F.interpolate(out, scale_factor = 2, mode='bilinear', align_corners=True)
        if select_inds is not None:
            return out, loss_distillation
        else:
            return out

from typing import Sequence, Dict, Optional
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = (6, 12, 18)) -> None:
        super(DeepLabHead, self).__init__()
        self.classifier = nn.Sequential(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

