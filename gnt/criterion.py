import torch.nn as nn
from utils import img2mse


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log


class SemanticCriterion(nn.Module):

    def __init__(self, args):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category

        """
        super().__init__()

        self.ignore_label = args.ignore_label
        self.num_classes = args.num_classes + 1 # +1 for ignore

    def compute_label_loss(self, label_pr, label_gt):
        label_pr = label_pr.reshape(-1, self.num_classes)
        label_gt = label_gt.reshape(-1).long()
        valid_mask = (label_gt != self.ignore_label)
        label_pr = label_pr[valid_mask]
        label_gt = label_gt[valid_mask]
        return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)


    def forward(self, outputs, ray_batch, scalars_to_log):
        pred_rgb, pred_label = outputs["rgb"], outputs["sems"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb, gt_label = ray_batch["rgb"], ray_batch["labels"]

        rgb_loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        label_loss = self.compute_label_loss(pred_label, gt_label)

        return rgb_loss, label_loss, scalars_to_log
