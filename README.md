
python -m torch.distributed.launch --nproc_per_node=8 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       ft_scannet.py --config configs/gnt_scannet_ft.txt --expname ft_depth_gpu_8



export CUDA_VISIBLE_DEVICES=0,1,6,7
python -m torch.distributed.launch --nproc_per_node=4 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/7_gpu_0903_2/model_239999.pth --expname distill_gpu_4 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler



python -m torch.distributed.launch --nproc_per_node=6 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/7_gpu_0903_2/model_stage_2.pth --expname distill_gpu_4 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler



python -m torch.distributed.launch --nproc_per_node=8 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/7_gpu_0903_2/model_stage_2.pth --expname distill_depth_gpu_8 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler