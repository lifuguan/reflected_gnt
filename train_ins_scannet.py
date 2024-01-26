import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
from torch.nn import functional as F

from gnt.data_loaders import dataset_dict
from gnt.render_ray import render_rays
from gnt.render_image import render_single_image
from gnt.model import GNTModel


from gnt.sample_ray import RaySamplerSingleImage
from utils import img_HWC2CHW, img2psnr, colorize, img2psnr, lpips, ssim
from gnt.loss import RenderLoss, SemanticLoss, InsEvaluator, DepthLoss
import config
import torch.distributed as dist
from gnt.projection import Projector
import imageio, cv2
import wandb 
import h5py

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank=0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

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


def train(args):

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

    # create finetuning dataset for each scene
    train_set_lists, val_set_lists, scene_set_names= [], [], []
    ft_scenes = np.loadtxt(args.val_set_list, dtype=str).tolist()
    for name in ft_scenes:
        train_dataset = dataset_dict['instance_replica'](args, is_train=True, scenes=name)
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if args.distributed
            else None
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            shuffle=True if train_sampler is None else False,
        )
        train_set_lists.append(train_loader)
        val_dataset = dataset_dict['instance_replica'](args, is_train=False, scenes=name)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
        val_set_lists.append(val_loader)
        scene_set_names.append(name)
        os.makedirs(out_folder + '/' + name, exist_ok=True)
        print(f'{name} val set len {len(val_loader)}')

    # create projector
    projector = Projector(device=device)

    # Create criterion
    render_criterion = RenderLoss(args)
    depth_criterion = DepthLoss(args)
    scalars_to_log = {}

    all_ap75_scores = {k:0 for k in scene_set_names}
    import json
    gt_color_dict_path = './data/Replica_DM/color_dict.json'
    gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
    
    for train_loader, val_loader, scene_name in zip(train_set_lists, val_set_lists, scene_set_names):
        color_dict = gt_color_dict['replica'][scene_name]
        color_f = os.path.join(args.rootdir, 'data/Replica_DM', scene_name, 'ins_rgb.hdf5')
        with h5py.File(color_f, 'r') as f:
            args.semantic_color_map = f['datasets'][:]
            args.num_classes = len(args.semantic_color_map)
        f.close()
        ins_evaluator = InsEvaluator(ins_num = args.num_classes)
        # Create GNT model
        model = GNTModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)

        epoch, global_step = 0, model.start_step + 1
        while global_step < model.start_step + args.total_step + 1:
            for train_data in train_loader:
                time0 = time.time()

                if args.distributed:
                    train_sampler.set_epoch(epoch)

                # load training rays
                ray_sampler = RaySamplerSingleImage(train_data, device)
                N_rand = int(
                    1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
                )
                ray_batch = ray_sampler.random_sample(
                    N_rand,
                    sample_mode=args.sample_mode,
                    center_ratio=args.center_ratio,
                )

                # reference feature extractor
                ref_coarse_feats, _, ref_deep_semantics = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
                ref_deep_semantics = model.feature_fpn(ref_deep_semantics)

                # novel view feature extractor
                _, _, que_deep_semantics = model.feature_net(train_data["rgb"].permute(0, 3, 1, 2).to(device))
                que_deep_semantics = model.feature_fpn(que_deep_semantics)

                ret = render_rays(
                    ray_batch=ray_batch,
                    model=model,
                    projector=projector,
                    featmaps=ref_coarse_feats,
                    ref_deep_semantics=ref_deep_semantics.detach(), # reference encoder的语义输出
                    # ref_deep_semantics=ref_deep_semantics, # reference encoder的语义输出
                    N_samples=args.N_samples,
                    inv_uniform=args.inv_uniform,
                    N_importance=args.N_importance,
                    det=args.det,
                    white_bkgd=args.white_bkgd,
                    ret_alpha=args.N_importance > 0,
                    single_net=args.single_net,
                    save_feature=args.save_feature,
                    model_type = args.model
                )

                selected_inds = ray_batch["selected_inds"]
                corase_sem_out, loss_distill = model.sem_seg_head(que_deep_semantics, ret['outputs_fine']['feats_out'], selected_inds)
                corase_sem_out = torch.sigmoid(corase_sem_out.permute(0,2,3,1))

                del ret['outputs_coarse']['feats_out'], ret['outputs_fine']['feats_out']
                ret['outputs_coarse']['sems'] = corase_sem_out
                ret['outputs_fine']['sems'] = corase_sem_out

                ray_batch['labels'] = train_data['labels'].to(device)

                # compute loss
                render_loss = render_criterion(ret, ray_batch)
                depth_loss = depth_criterion(ret, ray_batch)
                ins_loss_coarse, valid_ce_coarse, invalid_ce_coarse, valid_siou_coarse  = \
                    ins_evaluator.ins_criterion(ret['outputs_coarse']['sems'], ray_batch['labels'])
                ins_loss_fine, valid_ce_coarse, invalid_ce_coarse, valid_siou_coarse  = \
                    ins_evaluator.ins_criterion(ret['outputs_fine']['sems'], ray_batch['labels'])
                loss = (ins_loss_coarse+ins_loss_fine) * args.semantic_loss_scale + render_loss['train/rgb-loss'] + \
                       loss_distill * args.distill_loss_scale + depth_loss['train/depth-loss']

                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                model.scheduler.step()

                scalars_to_log["loss"] = loss.item()
                scalars_to_log["train/instance-loss"] = ins_loss_fine.item()
                scalars_to_log["train/rgb-loss"] = render_loss['train/rgb-loss'].item()
                scalars_to_log["train/depth-loss"] = depth_loss['train/depth-loss'].item()

                scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
                # end of core optimization loop
                dt = time.time() - time0

                # Rest is logging
                if args.rank == 0:
                    if global_step % args.i_print == 0 or global_step < 10:
                        # write psnr stats
                        psnr_metric = img2psnr(ret["outputs_coarse"]["rgb"], ray_batch["rgb"]).item()
                        scalars_to_log["train/coarse-psnr"] = psnr_metric
                        
                        pred_label, ap, pred_matched_order, gt_label_np = ins_evaluator.ins_eval(ret['outputs_fine']['sems'], ray_batch['labels'])

                        ins_map = {}
                        for idx, pred_label_replica in enumerate(pred_matched_order):
                            if pred_label_replica != -1:
                                ins_map[str(pred_label_replica)] = int(gt_label_np[idx])

                        scalars_to_log["train/AP50"] = ap[0]
                        scalars_to_log["train/AP75"] = ap[1]
                        scalars_to_log["train/AP95"] = ap[-1]

                        pred_ins_img = ins_evaluator.render_label2img(pred_label, args.semantic_color_map, color_dict, ins_map)
                        gt_ins_img = ins_evaluator.render_gt_label2img(ray_batch['labels'][0], args.semantic_color_map, color_dict)

                        logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                        for k in scalars_to_log.keys():
                            logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                        print(logstr)

                        if args.expname != 'debug':
                            wandb.log({
                            'images': wandb.Image(train_data["rgb"][0].cpu().numpy()),
                            'masks': {
                                'true': wandb.Image(gt_ins_img),
                                'pred': wandb.Image(pred_ins_img),
                            }})
                        del ray_batch

                    if args.expname != 'debug':
                        wandb.log(scalars_to_log)

                    if (global_step+1) % args.save_interval == 0:
                    # if (global_step+1) % 100 == 0:
                        print("Evaluating...")
                        indx = 0
                        psnr_scores,lpips_scores,ssim_scores, depth_scores = [],[],[],[]
                        ap50,ap75,ap95,que_ap50,que_ap75,que_ap95=[],[],[],[],[],[]
                        for val_data in val_loader:
                            tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
                            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                            gt_depth = val_data['true_depth'][0]

                            psnr_curr_img, lpips_curr_img, ssim_curr_img, ap, que_ap = log_view(
                                indx,
                                args,
                                model,
                                tmp_ray_sampler,
                                projector,
                                gt_img,
                                gt_depth,
                                evaluator=[ins_evaluator, depth_criterion],
                                render_stride=args.render_stride,
                                prefix="val/",
                                out_folder=out_folder,
                                ret_alpha=args.N_importance > 0,
                                single_net=args.single_net,
                                val_name = scene_name,
                                color_dict = color_dict
                            )
                            psnr_scores.append(psnr_curr_img)
                            lpips_scores.append(lpips_curr_img)
                            ssim_scores.append(ssim_curr_img)
                            ap50.append(ap[0])
                            ap75.append(ap[1])
                            ap95.append(ap[-1])
                            que_ap50.append(que_ap[0])
                            que_ap75.append(que_ap[1])
                            que_ap95.append(que_ap[-1])
                            torch.cuda.empty_cache()
                            indx += 1
                        scene_psnr  = np.mean(psnr_scores)
                        scene_psnr = np.mean(psnr_scores)
                        scene_lpips = np.mean(lpips_scores)
                        scene_ssim = np.mean(ssim_scores)
                        scene_ap50  = np.mean(ap50)
                        scene_ap75  = np.mean(ap75)
                        scene_ap95  = np.mean(ap95)
                        que_scene_ap50  = np.mean(que_ap50)
                        que_scene_ap75  = np.mean(que_ap75)
                        que_scene_ap95  = np.mean(que_ap95)
                        print("Average {} PSNR: {:.2f}, LPIPS: {:.2f}, SSIM: {:.2f}, AP50: {:.2f}, AP75: {:.2f}, AP95: {:.2f}, que_AP50: {:.2f}, que_AP75: {:.2f}, que_AP95: {:.2f}".format(
                            scene_name,scene_psnr,scene_lpips,scene_ssim, \
                                scene_ap50,scene_ap75, scene_ap95, que_scene_ap50,que_scene_ap75, que_scene_ap95))
                        wandb.log({"val-PSNR/{}".format(scene_name): scene_psnr,
                                    "val-LPIPS/{}".format(scene_name): scene_lpips,
                                    "val-SSIM/{}".format(scene_name): scene_ssim,
                                    "val-AP50/{}".format(scene_name): scene_ap50,
                                    "val-AP75/{}".format(scene_name): scene_ap75,
                                    "val-AP95/{}".format(scene_name): scene_ap95,
                                    "val-que_AP50/{}".format(scene_name): que_scene_ap50,
                                    "val-que_AP75/{}".format(scene_name): que_scene_ap75,
                                    "val-que_AP95/{}".format(scene_name): que_scene_ap95})
                        
                        # 如果比上一次的miou大，则
                        if scene_ap75 > all_ap75_scores[scene_name]:
                            all_ap75_scores[scene_name] = scene_ap75
                            print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                            fpath = os.path.join(out_folder, "best_{}.pth".format(scene_name))
                            model.save_model(fpath)
                 
                global_step += 1
                if global_step > model.start_step + args.total_step + 1:
                    break
            epoch += 1
    if args.expname != 'debug':
        print("All Scenes best IoU results: {}".format(all_ap75_scores))
        wandb.log(all_ap75_scores) # 输出所有的最优iou
        values = all_ap75_scores.values()
        mean_iou = sum(values) / len(values)
        print("Average IoU result: {}".format(mean_iou))
        wandb.log({"Average IoU":mean_iou})

@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img_vanilla,
    gt_depth,
    evaluator,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    single_net=True,
    val_name = None,
    color_dict = None
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()

        ref_coarse_feats, _, ref_deep_semantics = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        ref_deep_semantics = model.feature_fpn(ref_deep_semantics)
        device = ref_deep_semantics.device

        _, _, que_deep_semantics = model.feature_net(gt_img_vanilla.unsqueeze(0).permute(0, 3, 1, 2).to(ref_coarse_feats.device))
        que_deep_semantics = model.feature_fpn(que_deep_semantics)
        
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
        corase_sem_out = model.sem_seg_head(ret['outputs_coarse']['feats_out'][::2,::2,:].permute(2,0,1).unsqueeze(0).to(device), None, None)
        fine_sem_out = model.sem_seg_head(ret['outputs_fine']['feats_out'][::2,::2,:].permute(2,0,1).unsqueeze(0).to(device), None, None)
        ret['outputs_coarse']['sems'] = torch.sigmoid(corase_sem_out.permute(0,2,3,1))
        ret['outputs_fine']['sems'] = torch.sigmoid(fine_sem_out.permute(0,2,3,1))
        
        ret['que_sems'] = model.sem_seg_head(que_deep_semantics, None, None).permute(0,2,3,1)
        


    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))
    if args.render_stride != 1:
        gt_img = gt_img_vanilla[::render_stride, ::render_stride]
        gt_depth = gt_depth[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]
    else:
        gt_img = gt_img_vanilla

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
        depth_pred = torch.cat((colorize(gt_depth.squeeze(-1).detach().cpu(), cmap_name="jet", append_cbar=True, cbar_in_image=True), colorize(depth_pred, cmap_name="jet", append_cbar=True, cbar_in_image=True)), dim=1)

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
    filename = os.path.join(out_folder, val_name, "rgb_{:03d}.png".format(global_step))
    imageio.imwrite(filename, rgb_im)
    if depth_im is not None:
        depth_im = depth_im.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(out_folder, val_name, "depth_{:03d}.png".format(global_step))
        imageio.imwrite(filename, depth_im)
    
    try:
        if args.expname != 'debug':
            wandb.log({'val-depth_img': wandb.Image(depth_im)})
    except:
        pass

    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None else ret["outputs_coarse"]["rgb"]
    )

    lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    ssim_curr_img = ssim(pred_rgb, gt_img, format="HWC").item()
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    pred_label, ap, pred_matched_order, gt_label_np = evaluator[0].ins_eval(ret['outputs_fine']['sems'], ray_batch['labels'].reshape(h_max, w_max).unsqueeze(0))

    ins_map = {}
    for idx, pred_label_replica in enumerate(pred_matched_order):
        if pred_label_replica != -1:
            ins_map[str(pred_label_replica)] = int(gt_label_np[idx])

    pred_ins_img = evaluator[0].render_label2img(pred_label, args.semantic_color_map, color_dict, ins_map)

    que_pred_label, que_ap, que_pred_matched_order, que_gt_label_np = evaluator[0].ins_eval(ret['que_sems'], ray_batch['labels'].reshape(h_max, w_max).unsqueeze(0))
    que_ins_map = {}
    for idx, pred_label_replica in enumerate(que_pred_matched_order):
        if pred_label_replica != -1:
            que_ins_map[str(pred_label_replica)] = int(que_gt_label_np[idx])
    que_pred_ins_img = evaluator[0].render_label2img(que_pred_label, args.semantic_color_map, color_dict, que_ins_map)

    gt_ins_img = evaluator[0].render_gt_label2img(ray_batch['labels'].reshape(h_max, w_max), args.semantic_color_map, color_dict)
    ins_img_save = np.concatenate([gt_img_vanilla.numpy() * 255,gt_ins_img,pred_ins_img,que_pred_ins_img], axis=1)
    filename = os.path.join(out_folder, val_name, "ins_{:03d}.png".format(global_step))
    cv2.imwrite(filename, ins_img_save)


    print(prefix + "psnr_image: ", psnr_curr_img)
    print(prefix + "lpips_image: ", lpips_curr_img)
    print(prefix + "ssim_image: ", ssim_curr_img)
    print(prefix + "AP50: ", ap[0])
    print(prefix + "AP75: ", ap[1])
    print(prefix + "AP95: ", ap[-1])
    print(prefix + "que_AP50: ", que_ap[0])
    print(prefix + "que_AP75: ", que_ap[1])
    print(prefix + "que_AP95: ", que_ap[-1])
    model.switch_to_train()
    return psnr_curr_img, lpips_curr_img, ssim_curr_img, ap, que_ap

if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()
    
    init_distributed_mode(args)
    if args.rank == 0 and args.expname != 'debug':
        wandb.init(
            # set the wandb project where this run will be logged
            entity="vio-research",
            project="Semantic-NeRF",
            name=args.expname,
            
            # track hyperparameters and run metadata
            config={
            "N_samples": args.N_samples,
            "N_importance": args.N_importance,
            "chunk_size": args.chunk_size,
            "N_rand": args.N_rand,
            "semantic_loss_scale": args.semantic_loss_scale,
            "render_loss_scale": args.render_loss_scale,
            "lrate_semantic": args.lrate_semantic,
            "lrate_gnt": args.lrate_gnt,
            }
        )

    train(args)