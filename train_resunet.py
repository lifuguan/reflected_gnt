import os
import time
import numpy as np
import yaml
import shutil
import logging
import torch
import torch.utils.data.distributed
from torch.nn import functional as F
import torch.nn as nn
from pathlib import Path
import  wandb
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings

from gnt.data_loaders import dataset_dict
from gnt.feature_network import ResUNetLight
from gnt.model import OnlySemanticModel, SSLSemModel
from torch import optim
from gnt.criterion import SemanticCriterion
import config

from sklearn.metrics import confusion_matrix

from gnt.data_loaders.semantic_dataset import RandomRendererDataset, OrderRendererDataset

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

class SemanticLoss(Loss):
    def __init__(self, cfg):
        super().__init__(['loss_semantic'])
        self.cfg = cfg
        self.ignore_label = cfg['ignore_label']

    def __call__(self, data_pr, data_gt, step, **kwargs):
        num_classes = data_pr['pixel_label_nr'].shape[-1]
        def compute_loss(label_pr, label_gt):
            label_pr = label_pr.reshape(-1, num_classes)
            label_gt = label_gt.reshape(-1).long()
            valid_mask = (label_gt != self.ignore_label)
            label_pr = label_pr[valid_mask]
            label_gt = label_gt[valid_mask]
            return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)
        
        pixel_label_gt = data_pr['pixel_label_gt']
        pixel_label_nr = data_pr['pixel_label_nr']
        coarse_loss = compute_loss(pixel_label_nr, pixel_label_gt)
        
        if 'pixel_label_gt_fine' in data_pr:
            pixel_label_gt_fine = data_pr['pixel_label_gt_fine']
            pixel_label_nr_fine = data_pr['pixel_label_nr_fine']
            fine_loss = compute_loss(pixel_label_nr_fine, pixel_label_gt_fine)
        else:
            fine_loss = torch.zeros_like(coarse_loss)
        
        loss = (coarse_loss + fine_loss) * self.cfg['semantic_loss_scale']
        
        if 'pred_labels' in data_pr:
            ref_labels_pr = data_pr['pred_labels'].permute(0, 2, 3, 1)
            ref_labels_gt = data_gt['ref_imgs_info']['labels'].permute(0, 2, 3, 1)
            ref_loss = compute_loss(ref_labels_pr, ref_labels_gt)
            loss += ref_loss * self.cfg['semantic_loss_scale']
        return {'loss_semantic': loss}


# From https://github.com/Harry-Zhi/semantic_nerf/blob/a0113bb08dc6499187c7c48c3f784c2764b8abf1/SSR/training/training_utils.py
class IoU(Loss):
    default_cfg = {
        'ignore_label': 20,
        'num_classes': 20,
    }

    def __init__(self, cfg):
        super().__init__([])
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'pixel_label_nr_fine' in data_pr:
            true_labels = data_pr['pixel_label_gt_fine'].reshape(
                [-1]).long().detach().cpu().numpy()
            predicted_labels = data_pr['pixel_label_nr_fine'].argmax(
                dim=-1).reshape([-1]).long().detach().cpu().numpy()
        else:
            true_labels = data_pr['pixel_label_gt'].reshape(
                [-1]).long().detach().cpu().numpy()
            predicted_labels = data_pr['pixel_label_nr'].argmax(
                dim=-1).reshape([-1]).long().detach().cpu().numpy()

        if self.cfg['ignore_label'] != -1:
            valid_pix_ids = true_labels != self.cfg['ignore_label']
        else:
            valid_pix_ids = np.ones_like(true_labels, dtype=bool)

        num_classes = self.cfg['num_classes']
        predicted_labels = predicted_labels[valid_pix_ids]
        true_labels = true_labels[valid_pix_ids]

        conf_mat = confusion_matrix(
            true_labels, predicted_labels, labels=list(range(num_classes)))
        norm_conf_mat = np.transpose(np.transpose(
            conf_mat) / conf_mat.astype(float).sum(axis=1))

        # missing class will have NaN at corresponding class
        missing_class_mask = np.isnan(norm_conf_mat.sum(1))
        exsiting_class_mask = ~ missing_class_mask

        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        ious = np.zeros(num_classes)
        for class_id in range(num_classes):
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



@torch.inference_mode()
def evaluate(net, val_set_names, val_loader_list, device, amp, losses, args):
    net.eval()

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):

        all_eval_results = {}
        for (val_loader, val_name) in zip(val_loader_list, val_set_names):
            eval_results = {}
            num_val_batches = len(val_loader)
            for batch in tqdm(val_loader, total=num_val_batches, desc=val_name, unit='batch', leave=False):
                images, mask_true = batch['rgb'], batch['labels']

                # move images and labels to correct device and type
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                for loss in losses:
                    # predict the mask
                    if args.model == 'resunet':
                        masks_pred,_,_ = net(images.permute(0,3,1,2)) # ResUNetLighting
                        masks_pred = F.interpolate(
                        masks_pred, size=(240, 320), mode="bilinear", align_corners=False
                        ).permute(0,2,3,1)
                        loss_results=loss({"pixel_label_nr":masks_pred, "pixel_label_gt":mask_true}, None, None)
                    elif args.model == 'semanticfpn':
                        masks_pred = net(images.permute(0,3,1,2))
                        masks_pred = F.interpolate(
                        masks_pred, size=(240, 320), mode="bilinear", align_corners=False
                        ).permute(0,2,3,1)
                        loss_results=loss({"pixel_label_nr":masks_pred, "pixel_label_gt":mask_true}, None, None)
                    elif args.model == 'SSLSemModel':
                        images = F.interpolate(images.permute(0,3,1,2), scale_factor = 2, mode='bilinear', align_corners=True) # 先扩展一倍
                        masks_pred = net(images)
                        masks_pred = F.interpolate(
                        masks_pred, size=(240, 320), mode="bilinear", align_corners=False
                        ).permute(0,2,3,1)
                        loss_results=loss({"pixel_label_nr":masks_pred, "pixel_label_gt":   mask_true}, None, None)

                    else:
                        batch_inputs = []
                        for i in range(images.shape[0]):
                            image = images[i]
                            batch_inputs.append({"image": image.permute(2,0,1)})
                        masks_pred = net(batch_inputs)
                        masks_pred['pixel_label_gt'] = mask_true
                        loss_results=loss(masks_pred,None, None)
               
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                    for k,v in loss_results.items():
                        if type(v)==torch.Tensor:
                            v=v.detach().cpu().numpy()

                        if k in eval_results:
                            eval_results[k].append(v.item())
                        else:
                            eval_results[k]=[v.item()]
            for k in eval_results:
                if k in all_eval_results:
                    all_eval_results[k].append(np.mean(eval_results[k]))
                else:
                    all_eval_results[k] = [np.mean(eval_results[k])]

    for k,v in all_eval_results.items():
        if np.isscalar(v):
            v = np.expand_dims(v, axis=0)
        all_eval_results[k]=np.mean(v)
    net.train()
    return all_eval_results


def semantic_branch_setup(semantic_config_file):
    """
    Create configs and perform basic setups.
    """
    semantic_cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(semantic_cfg)
    add_mask_former_config(semantic_cfg)
    semantic_cfg.merge_from_file(semantic_config_file)
    semantic_cfg.merge_from_list([])
    semantic_cfg.freeze()
    # Setup logger for "mask_former" module
    return semantic_cfg
    
def train_model(
        load,
        device,
        iters: int = 5,
        batch_size: int = 16,
        args = None,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        wandb_name='ResUNet',
):
    train_set = RandomRendererDataset(is_train=True)
    # train_set = OrderRendererDataset(is_train=True)
    # create validation dataset
    val_set_lists, val_set_names = [], []
    val_scenes = np.loadtxt(args.val_set_list, dtype=str).tolist()
    for name in val_scenes:
        val_dataset = dataset_dict['val_scannet'](args, is_train=False, scenes=name)
        val_loader = DataLoader(val_dataset, batch_size=1)
        val_set_lists.append(val_loader)
        val_set_names.append(name)
        print(f'{name} val set len {len(val_loader)}')

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=64, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)

    if args.model == 'resunet':
        semantic_model = ResUNetLight(out_dim=20+1)
    elif args.model == 'semanticfpn':
        semantic_model = OnlySemanticModel(args)
    elif args.model == 'SSLSemModel':
        semantic_model = SSLSemModel(args)
    else:
        pass
    
    if load:
        state_dict = torch.load(load, map_location=device)
        semantic_model.load_state_dict(state_dict)
        print(f'Model loaded from {load}')
    if args.backbone_pretrain:
            print("Loading backbone pretrain model from : ", args.backbone_pretrain)
            state_dict = torch.load(args.backbone_pretrain, map_location=device)
            semantic_model.feature_net.load_state_dict(state_dict, strict=False)
            # for param in semantic_model.feature_net.parameters():
            #     param.requires_grad = False   
    semantic_model.to(device=device)

    optimizer = optim.Adam(semantic_model.parameters(),
                              lr=1e-3, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

    plotter = SemanticCriterion(args)
    criterion = SemanticLoss({"ignore_label":20, "semantic_loss_scale": 0.25})
    evaluator = IoU({"ignore_label":20, "semantic_loss_scale": 0.25})
    # 5. Begin training
    semantic_model.train()
    global_step = 0
    iters_loss = 0

    if args.expname != 'debug':
        experiment = wandb.init(entity="lifuguan",project="General-NeRF",name=args.expname)
    
    while global_step < iters: 
        for train_data in train_loader: 
            images, true_masks = train_data['image'], train_data['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last).permute(0,3,1,2)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            if args.model == 'resunet':
                masks_pred,_,_ = semantic_model(images) # ResUNetLighting
                masks_pred = F.interpolate(
                    masks_pred, size=(240, 320), mode="bilinear", align_corners=False).permute(0,2,3,1)
                loss_semantic = criterion({"pixel_label_nr":masks_pred, "pixel_label_gt":true_masks},None, global_step)
                loss = loss_semantic['loss_semantic']
            elif args.model == 'semanticfpn':
                masks_pred = semantic_model(images).permute(0,2,3,1)  # OnlySemantic
                loss_semantic = criterion({"pixel_label_nr":masks_pred, "pixel_label_gt":true_masks},None, global_step)
                loss = loss_semantic['loss_semantic']
            elif args.model == 'SSLSemModel':
                images = F.interpolate(images, scale_factor = 2, mode='bilinear', align_corners=True) # 先扩展一倍
                masks_pred = semantic_model(images).permute(0,2,3,1)  # OnlySemantic
                loss_semantic = criterion({"pixel_label_nr":masks_pred, "pixel_label_gt":true_masks},None, global_step)
                loss = loss_semantic['loss_semantic']
            else:                
                batch_inputs = []
                for i in range(images.shape[0]):
                    image = images[i]
                    true_mask = true_masks[i]
                    instances = Instances((240, 320))
                    classes = true_mask.unique()
                    classes = classes[classes != 20]
                    masks = []
                    for class_id in classes:
                        masks.append(true_mask == class_id)
                    instances.gt_classes = classes
                    if len(masks) == 0:
                        # Some image does not have annotation (all ignored)
                        instances.gt_masks = torch.zeros((0, true_mask.shape[-2], true_mask.shape[-1]))
                    else:
                        instances.gt_masks = torch.stack(masks)
                    
                    batch_input = {
                        "image": image,
                        "sem_seg": true_mask,
                        "instances": instances
                    }  
                    batch_inputs.append(batch_input)
                semantic_output = semantic_model(batch_inputs)
                loss = 0
                for k, v in semantic_output.items():
                    loss = loss+torch.mean(v)



            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            iters_loss += loss.item()
            if args.model == 'maskformer':
                print('step: {}, loss: {}'.format(global_step, loss.item()))
                if args.expname != 'debug':
                    experiment.log({'train/loss': loss.item()})
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    iou_metric=evaluator({"pixel_label_nr":masks_pred, "pixel_label_gt":true_masks}, None, None)
                print('loss: {}   miou: {}'.format(loss.item(), iou_metric['miou'].item()))
                if args.expname != 'debug':
                    experiment.log({'train loss': loss.item(), 'train/iou':iou_metric['miou'].item()})
            if (global_step+1) % 2000 == 0:
                if args.batch_size == 1:
                    ray_batch = {"rgb": images, "sems": masks_pred, "labels": true_masks}
                    _ = plotter.plot_semantic_results(ray_batch, ray_batch, global_step)
                val_score = evaluate(semantic_model, val_set_names, val_set_lists, device, amp, [evaluator], args)
                if args.expname != 'debug':
                    experiment.log(val_score)
                    experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            # 'images': wandb.Image(images.cpu().numpy()),
                            # 'masks': {
                            #     'true': wandb.Image(true_masks[0].float().cpu().numpy()),
                            #     'pred': wandb.Image(masks_pred[0].argmax(dim=2).float().cpu().numpy())}
                                })

                dir_checkpoint = Path('./out/'+args.expname)
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = semantic_model.state_dict()
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_iter{}.pth'.format(global_step)))
                print(f'Checkpoint {global_step} saved!')
            
            if global_step == iters: break


def get_args():
    parser = argparse.ArgumentParser(description='Train the ResUNet on images and target masks')
    parser.add_argument('--iters', type=int, default=50000, help='Number of iters')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default='', help='Load model from a .pth file')
    parser.add_argument('--backbone_pretrain', type=str, default='', help='Load model from a .pth file')
    parser.add_argument('--name', type=str, default='Order', help='Load model from a .pth file')
    parser.add_argument('--model', type=str, default='resunet', help='')
    parser.add_argument('--expname', type=str, default='debug', help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=20, help='Number of classes')
    parser.add_argument('--num_classes', type=int, default=20, help='Number of classes')
    parser.add_argument('--ignore_label', type=int, default=20, help='Number of classes')

    parser.add_argument(
        "--coarse_feat_dim", type=int, default=32, help="2D feature dimension for coarse level"
    )
    parser.add_argument(
        "--fine_feat_dim", type=int, default=32, help="2D feature dimension for fine level"
    )
    parser.add_argument('--val_set_list', type=str, default="configs/scannetv2_test_split.txt")

    parser.add_argument('--selected_inds', action="store_true")

    parser.add_argument(
        "--lrate_decay_factor",
        type=float,
        default=0.6,
        help="decay learning rate by a factor every specified number of steps",
    )
    parser.add_argument(
        "--lrate_decay_steps",
        type=int,
        default=20000,
        help="decay learning rate by a factor every specified number of steps",
    )

    parser.add_argument(
        "--single_net",
        type=bool,
        default=True,
        help="use single network for both coarse and/or fine sampling",
    )
    parser.add_argument('--unbounded', action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.semantic_color_map=[
        [174, 199, 232],  # wall
        [152, 223, 138],  # floor
        [31, 119, 180],   # cabinet
        [255, 187, 120],  # bed
        [188, 189, 34],   # chair
        [140, 86, 75],    # sofa
        [255, 152, 150],  # table
        [214, 39, 40],    # door
        [197, 176, 213],  # window
        [148, 103, 189],  # bookshelf
        [196, 156, 148],  # picture
        [23, 190, 207],   # counter
        [247, 182, 210],  # desk
        [219, 219, 141],  # curtain
        [255, 127, 14],   # refrigerator
        [91, 163, 138],   # shower curtain
        [44, 160, 44],    # toilet
        [112, 128, 144],  # sink
        [227, 119, 194],  # bathtub
        [82, 84, 163],    # otherfurn
        [248, 166, 116]  # invalid
    ]
    args.num_source_views=10
    args.rectify_inplane_rotation=0.0
    args.rootdir='./'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    train_model(
        args = args,
        load=args.load,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        amp=args.amp,
        wandb_name=args.name
    )


