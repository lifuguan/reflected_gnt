import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mse2psnr

class RenderLoss(nn.Module):
    
    def __init__(self):
        self.render_loss_scale = 1
    def __call__(self, data_pred, data_gt, **kwargs):
        rgb_gt = data_gt["rgb"]  # 1,rn,3
        rgb_coarse = data_pred["outputs_coarse"]["rgb"]  # rn,3

        def compute_loss(rgb_pr, rgb_gt):
            loss = torch.sum((rgb_pr-rgb_gt)**2, -1)        # n
            loss = torch.mean(loss)
            return loss * self.render_loss_scale

        results = {"train/coarse-loss": compute_loss(rgb_coarse, rgb_gt)}
        # results = {"train/coarse-psnr-training-batch": mse2psnr(results["train/coarse-loss"])}

        if data_pred["outputs_fine"] is not None:
            rgb_fine = data_pred["outputs_fine"]["rgb"]  # 1,rn,3
            results = {"train/fine-loss": compute_loss(rgb_fine, rgb_gt)}
            # results = {"train/fine-psnr-training-batch": mse2psnr(results["train/fine-loss"])}
        return results
    

class DepthLoss(nn.Module):

    def __init__(self, cfg):
        self.depth_correct_thresh = 0.02
        self.depth_loss_type = 'l2'
        self.depth_loss_l1_beta = 0.05
        if self.depth_loss_type == 'smooth_l1':
            self.loss_op = nn.SmoothL1Loss(
                reduction='none', beta=self.args['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'true_depth' not in data_gt['ref_imgs_info']:
            return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        coords = data_pr['depth_coords']  # rfn,pn,2
        depth_pr = data_pr['depth_mean']  # rfn,pn
        depth_maps = data_gt['ref_imgs_info']['true_depth']  # rfn,1,h,w
        rfn, _, h, w = depth_maps.shape
        depth_gt = interpolate_feats(
            depth_maps, coords, h, w, padding_mode='border', align_corners=True)[..., 0]   # rfn,pn

        # transform to inverse depth coordinate
        depth_range = data_gt['ref_imgs_info']['depth_range']  # rfn,2
        near, far = -1/depth_range[:, 0:1], -1/depth_range[:, 1:2]  # rfn,1

        def process(depth):
            depth = torch.clamp(depth, min=1e-5)
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth
        depth_gt = process(depth_gt)

        # compute loss
        def compute_loss(depth_pr):
            if self.depth_loss_type == 'l2':
                loss = (depth_gt - depth_pr)**2
            elif self.depth_loss_type == 'smooth_l1':
                loss = self.loss_op(depth_gt, depth_pr)

            if data_gt['scene_name'].startswith('gso'):
                # rfn,1,h,w
                depth_maps_noise = data_gt['ref_imgs_info']['depth']
                depth_aug = interpolate_feats(
                    depth_maps_noise, coords, h, w, padding_mode='border', align_corners=True)[..., 0]  # rfn,pn
                depth_aug = process(depth_aug)
                mask = (torch.abs(depth_aug-depth_gt) <
                        self.depth_correct_thresh).float()
                loss = torch.sum(loss * mask, 1) / (torch.sum(mask, 1) + 1e-4)
            else:
                loss = torch.mean(loss, 1)
            return loss

        outputs = {'loss_depth': compute_loss(depth_pr)}
        if 'depth_mean_fine' in data_pr:
            outputs['loss_depth_fine'] = compute_loss(
                data_pr['depth_mean_fine'])
        return outputs
    



def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return:
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)    # [srn,1,n,2]
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)      # srn,f,n
    feats_inter = feats_inter.permute(0,2,1)
    return  feats_inter

