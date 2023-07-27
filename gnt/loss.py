import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import confusion_matrix
from skimage.io import imsave
from utils import concat_images_list
import numpy as np


def nanmean(data, **args):
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)

class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys = keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class RenderLoss(nn.Module):
    
    def __init__(self, args):
        self.render_loss_scale = args.render_loss_scale

    def compute_rgb_loss(self, rgb_pr, rgb_gt):
        loss = torch.sum((rgb_pr - rgb_gt) ** 2, -1)        # n
        loss = torch.mean(loss)
        return loss * self.render_loss_scale
        
    def __call__(self, data_pred, data_gt, **kwargs):
        rgb_gt = data_gt["rgb"]  # 1,rn,3
        rgb_coarse = data_pred["outputs_coarse"]["rgb"]  # rn,3

        results = {"train/rgb-loss": self.compute_rgb_loss(rgb_coarse, rgb_gt)}
        # results = {"train/coarse-psnr-training-batch": mse2psnr(results["train/coarse-loss"])}

        if data_pred["outputs_fine"] is not None:
            rgb_fine = data_pred["outputs_fine"]["rgb"]  # 1,rn,3
            results["train/rgb-loss"] += self.compute_rgb_loss(rgb_fine, rgb_gt)
            # results = {"train/fine-psnr-training-batch": mse2psnr(results["train/fine-loss"])}
        return results
    
class SemanticLoss(Loss):
    def __init__(self, args):
        super().__init__(['loss_semantic'])
        self.semantic_loss_scale = args.semantic_loss_scale
        self.ignore_label = args.ignore_label
        self.num_classes = args.num_classes + 1 # for ignore label
        self.color_map = torch.tensor(args.semantic_color_map, dtype=torch.uint8)
        self.expname = args.expname

    def plot_semantic_results(self, data_pred, data_gt, step):
        h, w = data_pred['sems'].shape[1:3]
        batch_size = data_pred['sems'].shape[0]
        self.color_map.to(data_gt['rgb'].device)
        
        def get_img(data_src, key, channel):
            rgbs = data_src[key]  # 1,rn,3
            rgbs = rgbs[0] if batch_size > 1 else rgbs
            rgbs = rgbs.reshape([h, w, channel]).detach()
            if channel > 1:
                rgbs = rgbs.argmax(axis=-1, keepdims=True)
            rgbs = rgbs.squeeze().cpu().numpy()
            rgbs = self.color_map[rgbs]
            return rgbs
        
        imgs = [get_img(data_gt, 'labels', 1), get_img(data_pred, 'sems', self.num_classes)]

        model_name = self.expname
        Path(f'out/vis/{model_name}').mkdir(exist_ok=True, parents=True)
        # imsave(f'out/vis/{model_name}/step-{step}-sem.png', concat_images_list(*imgs))
        return imgs
    
    def compute_semantic_loss(self, label_pr, label_gt, num_classes):
        label_pr = label_pr.reshape(-1, num_classes)
        label_gt = label_gt.reshape(-1).long()
        valid_mask = (label_gt != self.ignore_label)
        label_pr = label_pr[valid_mask]
        label_gt = label_gt[valid_mask]
        return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)
    
    def __call__(self, data_pred, data_gt, step, **kwargs):
        num_classes = data_pred['outputs_coarse']['sems'].shape[-1]
        
        pixel_label_gt = data_gt['labels']
        pixel_label_nr = data_pred['outputs_coarse']['sems']
        coarse_loss = self.compute_semantic_loss(pixel_label_nr, pixel_label_gt, num_classes)
        
        if 'outputs_fine' in data_pred:
            pixel_label_nr_fine = data_pred['outputs_fine']['sems']
            fine_loss = self.compute_semantic_loss(pixel_label_nr_fine, pixel_label_gt, num_classes)
        else:
            fine_loss = torch.zeros_like(coarse_loss)
        
        loss = (coarse_loss + fine_loss) * self.semantic_loss_scale
        
        if 'reference_sems' in data_pred:
            ref_labels_pr = data_pred['reference_sems']
            ref_labels_gt = data_gt['src_labels']
            ref_loss = self.compute_semantic_loss(ref_labels_pr, ref_labels_gt, num_classes)
            loss += ref_loss * self.semantic_loss_scale
        return {'train/semantic-loss': loss}

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
    
# From https://github.com/Harry-Zhi/semantic_nerf/blob/a0113bb08dc6499187c7c48c3f784c2764b8abf1/SSR/training/training_utils.py
class IoU(Loss):

    def __init__(self, args):
        super().__init__([])
        self.num_classes = args.num_classes
        self.ignore_label = args.ignore_label


    def __call__(self, data_pred, data_gt, step, **kwargs):
        true_labels = data_gt['labels'].reshape([-1]).long().detach().cpu().numpy()
        if 'outputs_fine' in data_pred:
            predicted_labels = data_pred['outputs_fine']['sems'].argmax(
                dim=-1).reshape([-1]).long().detach().cpu().numpy()
        else:
            predicted_labels = data_pred['outputs_coarse']['sems'].argmax(
                dim=-1).reshape([-1]).long().detach().cpu().numpy()

        if self.ignore_label != -1:
            valid_pix_ids = true_labels != self.ignore_label
        else:
            valid_pix_ids = np.ones_like(true_labels, dtype=bool)

        predicted_labels = predicted_labels[valid_pix_ids]
        true_labels = true_labels[valid_pix_ids]

        conf_mat = confusion_matrix(
            true_labels, predicted_labels, labels=list(range(self.num_classes)))
        norm_conf_mat = np.transpose(np.transpose(
            conf_mat) / conf_mat.astype(float).sum(axis=1))

        # missing class will have NaN at corresponding class
        missing_class_mask = np.isnan(norm_conf_mat.sum(1))
        exsiting_class_mask = ~ missing_class_mask

        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        ious = np.zeros(self.num_classes)
        for class_id in range(self.num_classes):
            ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
        miou = np.mean(ious[exsiting_class_mask])
        if np.isnan(miou):
            miou = 0.
            total_accuracy = 0.
            class_average_accuracy = 0.
        output = {
            'miou': torch.tensor([miou], dtype=torch.float32),
            'total_accuracy': torch.tensor([total_accuracy], dtype=torch.float32),
            'class_average_accuracy': torch.tensor([class_average_accuracy], dtype=torch.float32)
        }
        return output




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

