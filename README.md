
export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_fuxian1.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_fuxian1 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler





export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_smeantic_0831_1 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler

export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_smeantic_0831_2 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler

export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_smeantic_0831_3 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler


export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_smeantic_0831_4 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler


export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname que_agg --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler


CUDA_VISIBLE_DEVICES=6 python train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname one_gpu_0903_5 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler

CUDA_VISIBLE_DEVICES=7 python train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname one_gpu_0903_2 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler

CUDA_VISIBLE_DEVICES=5 python train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname one_gpu_0903_3 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler

CUDA_VISIBLE_DEVICES=4 python train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname one_gpu_0903_4 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler