export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_smeantic_0831_4/model_039999.pth --expname frozen_gnt --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler



export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_smeantic_0831_4/model_stage_2.pth --expname frozen_gnt2 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler

export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_smeantic_0831_4/model_stage_2.pth --expname frozen_gnt3 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler