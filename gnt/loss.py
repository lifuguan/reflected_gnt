import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import confusion_matrix
from skimage.io import imsave
from utils import concat_images_list
import numpy as np

from sklearn.decomposition import PCA
import sklearn
import time
from PIL import Image
import matplotlib.pyplot as plt

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
    

class DepthGuidedSemLoss(nn.Module):
    
    def __init__(self, args):
        self.dgs_loss_scale = args.dgs_loss_scale
        self.N_samples = args.N_samples
        
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

    def plot_pca_features(self, data_pred, ray_batch, step, val_name=None, vis=False):
        coarse_feats = data_pred['outputs_coarse']['feats_out'].unsqueeze(0).permute(0,3,1,2)
        fine_feats = data_pred['outputs_fine']['feats_out'].unsqueeze(0).permute(0,3,1,2)
        h, w = coarse_feats.shape[2:4]
        def pca_calc(feats):
            fmap = feats.cuda()
            pca = sklearn.decomposition.PCA(3, random_state=80)
            f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
            transformed = pca.fit_transform(f_samples)
            feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
            feature_pca_components = torch.tensor(pca.components_).float().cuda()
            q1, q99 = np.percentile(transformed, [1, 99])
            feature_pca_postprocess_sub = q1
            feature_pca_postprocess_div = (q99 - q1)
            del f_samples

            vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
            vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
            vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
            return (vis_feature.cpu().numpy() * 255).astype(np.uint8)

        rgbs = ray_batch['rgb']  # 1,rn,3
        rgbs = rgbs.reshape([h*2, w*2, 3]).detach() * 255
        rgbs = rgbs.squeeze().cpu().numpy().astype(np.uint8)[::2, ::2]     
        imgs = [rgbs, pca_calc(coarse_feats), pca_calc(fine_feats)]

        model_name = self.expname
        if vis is True:
            imsave(f'out/{model_name}/{val_name}/pca_{step}.png', concat_images_list(*imgs))

    def plot_semantic_results(self, data_pred, data_gt, step, val_name=None, vis=False):
        h, w = data_pred['sems'].shape[1:3]
        batch_size = data_pred['sems'].shape[0]
        self.color_map.to(data_gt['rgb'].device)
        
        if self.ignore_label != -1:
            unvalid_pix_ids = data_gt['labels'] == self.ignore_label
        else:
            unvalid_pix_ids = np.zeros_like(data_gt, dtype=bool)
        unvalid_pix_ids = unvalid_pix_ids.reshape(h,w,-1)

        def get_label_img(data_src, key, channel):
            rgbs = data_src[key]  # 1,rn,3
            rgbs = rgbs[0] if batch_size > 1 else rgbs
            rgbs = rgbs.reshape([h, w, channel]).detach()
            if channel > 1:
                rgbs = rgbs.argmax(axis=-1, keepdims=True)
                rgbs[unvalid_pix_ids] = self.ignore_label

            rgbs = rgbs.squeeze().cpu().numpy()
            rgbs = self.color_map[rgbs]
            return rgbs
        
        def get_rgb(data_src, key, channel):
            rgbs = data_src[key]  # 1,rn,3
            rgbs = rgbs.reshape([h, w, channel]).detach() * 255
            rgbs = rgbs.squeeze().cpu().numpy().astype(np.uint8)
            return rgbs
        
        if 'full_rgb' not in data_gt.keys():
            if 'que_sems' in data_pred.keys():
                imgs = [get_label_img(data_gt, 'labels', 1), get_label_img(data_pred, 'sems', self.num_classes), get_label_img(data_pred, 'que_sems', self.num_classes)]
            else:
                imgs = [get_label_img(data_gt, 'labels', 1), get_label_img(data_pred, 'sems', self.num_classes)]
        else:
            imgs = [get_rgb(data_gt, 'full_rgb', 3), get_label_img(data_gt, 'labels', 1), get_label_img(data_pred, 'sems', self.num_classes)]

        

        model_name = self.expname
        if vis is True:
            imsave(f'out/{model_name}/{val_name}/seg_{step}.png', concat_images_list(*imgs))
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
        
        # if 'pred_labels' in data_pred:
        #     ref_labels_pr = data_pred['pred_labels'].permute(0, 2, 3, 1)
        #     ref_labels_gt = data_gt['ref_imgs_info']['labels'].permute(0, 2, 3, 1)
        #     ref_loss = self.compute_semantic_loss(ref_labels_pr, ref_labels_gt, num_classes)
        #     loss += ref_loss * self.semantic_loss_scale
        return {'train/semantic-loss': loss}

class DepthLoss(nn.Module):

    def __init__(self, args):
        super(DepthLoss, self).__init__()
        self.depth_loss_scale = args.depth_loss_scale

        self.depth_correct_thresh = 0.02
        self.depth_loss_type = 'smooth_l1'
        self.depth_loss_l1_beta = 0.05
        self.loss_op = nn.SmoothL1Loss(reduction='none', beta=self.depth_loss_l1_beta)

    def __call__(self, data_pr, data_gt, **kwargs):
        depth_pr = data_pr['outputs_coarse']['depth']  # pn
        depth_gt = data_gt['true_depth']  # pn

        # transform to inverse depth coordinate
        depth_range = data_gt['depth_range']  # rfn,2
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
                loss = self.loss_op(depth_pr, depth_gt.squeeze(-1)) 

            return torch.mean(loss)

        outputs = {'train/depth-loss': compute_loss(depth_pr)}
        if 'outputs_fine' in data_pr:
            outputs['train/depth-loss'] += compute_loss(data_pr['outputs_fine']['depth'])
        outputs['train/depth-loss'] = outputs['train/depth-loss'] * self.depth_loss_scale
        return outputs
    
# From https://github.com/Harry-Zhi/semantic_nerf/blob/a0113bb08dc6499187c7c48c3f784c2764b8abf1/SSR/training/training_utils.py
class IoU(Loss):

    def __init__(self, args):
        super().__init__([])
        self.num_classes = args.num_classes
        self.ignore_label = args.ignore_label

    def iou_calc(self, predicted_labels, true_labels, valid_pix_ids):
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
        return miou, total_accuracy, class_average_accuracy
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

        miou, total_accuracy, class_average_accuracy = self.iou_calc(predicted_labels, true_labels, valid_pix_ids)
        
        if 'que_sems' in data_pred.keys():
            predicted_labels = data_pred['que_sems'].argmax(dim=-1).reshape([-1]).long().detach().cpu().numpy()
            que_miou, _, _ = self.iou_calc(predicted_labels, true_labels, valid_pix_ids)
            output = {
                'miou': torch.tensor([miou], dtype=torch.float32),
                'que_miou': torch.tensor([que_miou], dtype=torch.float32),
                'total_accuracy': torch.tensor([total_accuracy], dtype=torch.float32),
                'class_average_accuracy': torch.tensor([class_average_accuracy], dtype=torch.float32)
            }
        else:
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