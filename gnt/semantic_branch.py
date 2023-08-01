import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm



class NeRFSemSegFPNHead(nn.Module):

    def __init__(self, args, feature_strides=[2,4,8,16], feature_channels=[128,128,128,128], num_classes = 20):
        super(NeRFSemSegFPNHead, self).__init__()
        
        self.unbounded = args.unbounded # softmax or not
        self.selected_inds = args.selected_inds

        conv_dims = 128
        self.scale_heads = nn.ModuleList()
        self.common_stride = 2
        for stride, channels in zip(feature_strides, feature_channels):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                norm_module = get_norm('GN', conv_dims)
                conv = Conv2d(
                    channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not 'GN',
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if stride != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
        
        self.predictor = Conv2d(conv_dims, num_classes + 1, kernel_size=1, stride=1, padding=0)

        if args.unbounded is True:
            self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, deep_feats, out_feats, select_inds):
        h, w = deep_feats.shape[-2:]
        #######   replace feature map           #######
        if select_inds is not None:
            batch=1
            ratio = 240 * 320 // (deep_feats.shape[-2] * deep_feats.shape[-1])
            deep_feats = deep_feats.reshape(1, deep_feats.shape[1], -1).squeeze(0).permute(1,0)
            re_select_inds = torch.tensor([select_ind // ratio for select_ind in select_inds])
            deep_feats[re_select_inds] = out_feats
            chunks = torch.chunk(deep_feats, 4, dim=1)
        else:
            batch = deep_feats.shape[0]
            deep_feats = deep_feats.reshape(batch, deep_feats.shape[1], -1).permute(0,2,1)
            chunks = torch.chunk(deep_feats, 4, dim=2)

        ####### constrcut feature pyramids and Decoder  #######
        for i, chunk in enumerate(chunks):
            chunk = chunk.reshape(batch, h, w, chunk.shape[-1]).permute(0,3,1,2) # batch, 512, h, w
            if i == 0:
                x = self.scale_heads[i](chunk)
            else:
                chunk = F.interpolate(chunk, scale_factor = 1/(2**i), mode='bilinear', align_corners=True, recompute_scale_factor=True)
                x = x + self.scale_heads[i](chunk)

        out = self.predictor(x)
        out = F.interpolate(out, scale_factor = 240 // h, mode='bilinear', align_corners=True)  # b, c, h, w

        if self.selected_inds is True:
            out = out.reshape(batch, out.shape[1], -1)[:,:,re_select_inds]
        if self.unbounded is True:
            return self.softmax(out)
        else:
            return out 