import os
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
from torch.nn import functional as F

from torch.utils.data import DataLoader

from gnt.data_loaders import dataset_dict
from gnt.render_image import render_single_image
from gnt.model import GNTModel
from gnt.sample_ray import RaySamplerSingleImage
from utils import img_HWC2CHW, colorize, img2psnr, lpips, ssim
import config
import torch.distributed as dist
from gnt.projection import Projector
import imageio

from gnt.loss import SemanticLoss, IoU, DepthLoss
import logging


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@torch.no_grad()
def eval(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.CRITICAL,
                    filename=os.path.join(out_folder, "result.log"),
                    filemode='a')
    # create validation dataset
    val_set_lists, val_set_names = [], []
    val_scenes = np.loadtxt(args.val_set_list, dtype=str).tolist()
    for name in val_scenes:
        val_dataset = dataset_dict['val_scannet'](args, is_train=False, scenes=name)
        val_loader = DataLoader(val_dataset, batch_size=1)
        val_set_lists.append(val_loader)
        val_set_names.append(name)
        os.makedirs(out_folder + '/' + name.split('/')[1], exist_ok=True)
        print(f'{name} val set len {len(val_loader)}')

    # Create GNT model
    model = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    # create projector
    projector = Projector(device=device)

    iou_criterion = IoU(args)
    semantic_criterion = SemanticLoss(args)
    depth_criterion = DepthLoss(args)

    all_psnr_scores,all_lpips_scores,all_ssim_scores, all_iou_scores, all_depth_scores = [],[],[],[],[]
    for val_scene, val_name in zip(val_set_lists, val_set_names):
        indx = 0
        psnr_scores,lpips_scores,ssim_scores, iou_scores, depth_scores = [],[],[],[],[]
        for val_data in val_scene:
            tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
            gt_depth = val_data['true_depth'][0]

            psnr_curr_img, lpips_curr_img, ssim_curr_img, iou_metric, depth_metric = log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                projector,
                gt_img,
                gt_depth,
                evaluator=[iou_criterion, semantic_criterion, depth_criterion],
                render_stride=args.render_stride,
                prefix="val/" if args.run_val else "train/",
                out_folder=out_folder,
                ret_alpha=args.N_importance > 0,
                single_net=args.single_net,
                val_name = val_name.split('/')[1]
            )
            psnr_scores.append(psnr_curr_img)
            lpips_scores.append(lpips_curr_img)
            ssim_scores.append(ssim_curr_img)
            iou_scores.append(iou_metric)
            depth_scores.append(depth_metric)
            torch.cuda.empty_cache()
            indx += 1
        print("Average {} PSNR: {}, LPIPS: {}, SSIM: {}, IoU: {}, Depth: {}".format(
            val_name, 
            np.mean(psnr_scores),
            np.mean(lpips_scores),
            np.mean(ssim_scores),
            np.mean(iou_scores),
            np.mean(depth_scores)))
        all_psnr_scores.append(np.mean(psnr_scores))
        all_lpips_scores.append(np.mean(lpips_scores))
        all_ssim_scores.append(np.mean(ssim_scores))
        all_iou_scores.append(np.mean(iou_scores))    
        all_depth_scores.append(np.mean(depth_scores))
        logging.critical("Average {} PSNR: {}, LPIPS: {}, SSIM: {}, IoU: {}, Depth: {}".format(
            val_name, 
            np.mean(psnr_scores),
            np.mean(lpips_scores),
            np.mean(ssim_scores),
            np.mean(iou_scores),
            np.mean(depth_scores))) 
    print("Overall PSNR: {}, LPIPS: {}, SSIM: {}, IoU: {}, Depth: {}".format(
        np.mean(all_psnr_scores),
        np.mean(all_lpips_scores),
        np.mean(all_ssim_scores),
        np.mean(all_iou_scores),
        np.mean(all_depth_scores)))
    logging.critical("Overall PSNR: {}, LPIPS: {}, SSIM: {}, IoU: {}, Depth:{}".format(
        np.mean(all_psnr_scores),
        np.mean(all_lpips_scores),
        np.mean(all_ssim_scores),
        np.mean(all_iou_scores),
        np.mean(all_depth_scores)))


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    gt_depth,
    evaluator,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    single_net=True,
    val_name = None,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()

        ref_coarse_feats, _, ref_deep_semantics = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        ref_deep_semantics = model.feature_fpn(ref_deep_semantics)
        device = ref_coarse_feats.device
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=ref_coarse_feats,
            deep_semantics=ref_deep_semantics, # encoder的语义输出
            ret_alpha=ret_alpha,
            single_net=single_net,
        )

        ret['outputs_coarse']['sems'] = model.sem_seg_head(ret['outputs_coarse']['feats_out'][::2,::2,:].permute(2,0,1).unsqueeze(0).to(device), None, None).permute(0,2,3,1)
        ret['outputs_fine']['sems'] = model.sem_seg_head(ret['outputs_coarse']['feats_out'][::2,::2,:].permute(2,0,1).unsqueeze(0).to(device), None, None).permute(0,2,3,1)


    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))
    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        gt_depth = gt_depth[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    invalid_depth_mask = torch.ones_like(gt_depth)
    invalid_depth_mask[gt_depth == 0] = 0
    ray_batch['true_depth'] = gt_depth
    ray_batch['depth_range'] = ray_batch['depth_range'].to(gt_depth.device)

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3 * w_max)
    rgb_im[:, : average_im.shape[-2], : average_im.shape[-1]] = average_im
    rgb_im[:, : rgb_gt.shape[-2], w_max : w_max + rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2], 2 * w_max : 2 * w_max + rgb_pred.shape[-1]] = rgb_pred
    if "depth" in ret["outputs_coarse"].keys():
        depth_pred = ret["outputs_coarse"]["depth"].detach().cpu()
        depth_pred = torch.cat((colorize(gt_depth.squeeze(-1).detach().cpu(), cmap_name="jet", append_cbar=True, cbar_in_image=True), colorize(depth_pred, cmap_name="jet", cbar_in_image=True, append_cbar=True)), dim=1)

        depth_im = img_HWC2CHW(depth_pred)
    else:
        depth_im = None
    
    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret["outputs_fine"]["rgb"].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, : rgb_fine.shape[-2], : rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_pred = torch.cat((depth_pred, colorize(ret["outputs_fine"]["depth"].detach().cpu(), cmap_name="jet", append_cbar=True, cbar_in_image=True)), dim=1)
        depth_im = img_HWC2CHW(depth_pred)

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}.png".format(global_step))
    imageio.imwrite(filename, rgb_im)
    if depth_im is not None:
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, prefix[:-1] + "depth_{:03d}.png".format(global_step))
        imageio.imwrite(filename, depth_im)

    
    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None else ret["outputs_coarse"]["rgb"]
    )

    lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    ssim_curr_img = ssim(pred_rgb, gt_img, format="HWC").item()
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    iou_metric = evaluator[0](ret, ray_batch, global_step)
    sem_imgs = evaluator[1].plot_semantic_results(ret["outputs_coarse"], ray_batch, global_step, val_name, vis=True)
    depth_metric = evaluator[2](ret, ray_batch)

    print(prefix + "psnr_image: ", psnr_curr_img)
    print(prefix + "lpips_image: ", lpips_curr_img)
    print(prefix + "ssim_image: ", ssim_curr_img)
    print(prefix + "iou: ", iou_metric['miou'].item())
    print(prefix + "depth_loss: ", depth_metric['train/depth-loss'].item())
    model.switch_to_train()
    return psnr_curr_img, lpips_curr_img, ssim_curr_img, iou_metric['miou'].item(), depth_metric['train/depth-loss'].item()



if __name__ == "__main__":
    parser = config.config_parser()
    parser.add_argument("--run_val", action="store_true", help="run on val set")
    args = parser.parse_args()
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
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    eval(args)
