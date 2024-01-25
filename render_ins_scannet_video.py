import os
import cv2
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
from torch.nn import functional as F
from tqdm import tqdm

from gnt.data_loaders import dataset_dict
from gnt.render_ray import render_rays
from gnt.render_image import render_single_image
from gnt.model import GNTModel


from gnt.sample_ray import RaySamplerSingleImage
from utils import img_HWC2CHW, img2psnr, colorize, img2psnr, lpips, ssim
from gnt.loss import RenderLoss, SemanticLoss, InsEvaluator, DepthLoss, plot_pca_features
import config
import torch.distributed as dist
from gnt.projection import Projector
import h5py, logging, imageio


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
    render_set_lists, scene_set_names= [], []
    if args.train_scenes is not None:
        ft_scenes = args.train_scenes
    else:
        ft_scenes = np.loadtxt(args.val_set_list, dtype=str).tolist()
    for name in ft_scenes:
        render_dataset = dataset_dict['instance_replica'](args, is_train=True, scenes=name)
        render_loader = torch.utils.data.DataLoader(render_dataset, batch_size=1)
        render_set_lists.append(render_loader)
        scene_set_names.append(name)
        os.makedirs(out_folder + '/' + name, exist_ok=True)
        print(f'{name} render set len {len(render_loader)}')

    # create projector
    projector = Projector(device=device)

    # Create criterion
    depth_criterion = DepthLoss(args)

    all_ap75_scores = {k:0 for k in scene_set_names}
    import json
    gt_color_dict_path = './data/Replica_DM/color_dict.json'
    gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
    
    for render_loader, scene_name in zip(render_set_lists, scene_set_names):
        logging.basicConfig(format='%(message)s',
            level=logging.CRITICAL,
            filename=os.path.join(out_folder, scene_name, "result.log"),
            filemode='a')
        color_dict = gt_color_dict['replica'][scene_name]
        color_f = os.path.join(args.rootdir, 'data/Replica_DM', scene_name, 'ins_rgb.hdf5')
        with h5py.File(color_f, 'r') as f:
            args.semantic_color_map = f['datasets'][:]
            args.num_classes = len(args.semantic_color_map)
        f.close()
        ins_evaluator = InsEvaluator(ins_num = args.num_classes)
        args.ckpt_path = os.path.join(out_folder, f'best_{scene_name}.pth')
        # Create GNT model
        model = GNTModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)


        print("Evaluating...")
        indx = 0
        scene_rgb_fine=[]; scene_pred_ins_img=[]; scene_que_pred_ins_img=[]; scene_pca_img=[];        
        for render_data in tqdm(render_loader):
            tmp_ray_sampler = RaySamplerSingleImage(render_data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
            gt_depth = render_data['true_depth'][0]

            rgb_fine, pred_ins_img, que_pred_ins_img, pca_img = render_view(
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
            torch.cuda.empty_cache()
            scene_rgb_fine.append(rgb_fine); scene_pred_ins_img.append(pred_ins_img)
            scene_que_pred_ins_img.append(que_pred_ins_img); scene_pca_img.append(pca_img)
            indx += 1
        imageio.mimwrite(os.path.join(f'out/ins_replica_gpu_8/{scene_name}', 'rgb_fine.mp4'), scene_rgb_fine, fps=10, quality=8)
        imageio.mimwrite(os.path.join(f'out/ins_replica_gpu_8/{scene_name}', 'pred_ins_img.mp4'), scene_pred_ins_img, fps=10, quality=8)
        imageio.mimwrite(os.path.join(f'out/ins_replica_gpu_8/{scene_name}', 'que_pred_ins_img.mp4'), scene_que_pred_ins_img, fps=10, quality=8)
        imageio.mimwrite(os.path.join(f'out/ins_replica_gpu_8/{scene_name}', 'pca_img.mp4'), scene_pca_img, fps=10, quality=8)
    if args.expname != 'debug':
        print("All Scenes best IoU results: {}".format(all_ap75_scores))
        values = all_ap75_scores.values()
        mean_iou = sum(values) / len(values)
        print("Average IoU result: {}".format(mean_iou))

@torch.no_grad()
def render_view(
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
        
        if args.render_stride == 1:
            corase_sem_out = model.sem_seg_head(ret['outputs_coarse']['feats_out'][::2, ::2, :].permute(2,0,1).unsqueeze(0).to(device), None, None).permute(0,2,3,1)
            fine_sem_out = model.sem_seg_head(ret['outputs_fine']['feats_out'][::2, ::2, :].permute(2,0,1).unsqueeze(0).to(device), None, None).permute(0,2,3,1)
        else:
            corase_sem_out = model.sem_seg_head(ret['outputs_coarse']['feats_out'].permute(2,0,1).unsqueeze(0).to(device), None, None).permute(0,2,3,1)
            fine_sem_out = model.sem_seg_head(ret['outputs_fine']['feats_out'].permute(2,0,1).unsqueeze(0).to(device), None, None).permute(0,2,3,1)
        que_sem_out = model.sem_seg_head(que_deep_semantics, None, None)
        ret['outputs_coarse']['sems'] = F.softmax(corase_sem_out, dim=-1)
        ret['outputs_fine']['sems'] = F.softmax(fine_sem_out, dim=-1)
        
        
        ret['que_sems'] = F.softmax(que_sem_out.permute(0,2,3,1), dim=-1)


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

    rgb_gt = img_HWC2CHW(gt_img)
    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    if ret["outputs_fine"] is not None:
        rgb_fine = ret["outputs_fine"]["rgb"].detach().cpu()

    # write scalar
    pred_rgb = (
        ret["outputs_fine"]["rgb"]
        if ret["outputs_fine"] is not None else ret["outputs_coarse"]["rgb"]
    )

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

    pca_img = plot_pca_features(ret, ray_batch, global_step, val_name, vis=True, return_img = True)

    model.switch_to_train()
    return rgb_fine, pred_ins_img, que_pred_ins_img, pca_img

if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()
    
    init_distributed_mode(args)
    train(args)