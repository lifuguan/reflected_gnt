
export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_fuxian1.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_fuxian1 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler





export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_smeantic_0817_1 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler

export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_smeantic_0817_2 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler

export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_smeantic_0817_3 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler


export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_smeantic_0817_4 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler