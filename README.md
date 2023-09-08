export CUDA_VISIBLE_DEVICES=5,6
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_smeantic_0831_4/model_stage_2.pth --expname distill1 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler


export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_smeantic_0831_4/model_stage_2.pth --N_rand 1024 --expname distill_more_ray --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler




export CUDA_VISIBLE_DEVICES=2,3,4,5
python -m torch.distributed.launch --nproc_per_node=4 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/7_gpu_0903_2/model_stage_2.pth --expname no_agg_deatch_gpu_4 --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler


export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname no_agg_deatch_gpu_4_scratch --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler