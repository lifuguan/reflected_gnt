python3 eval.py --config configs/gnt_full.txt --eval_scenes orchids --expname gnt_full --chunk_size 500 --run_val --N_samples 192
python3 eval.py --config configs/gnt_llff.txt --eval_scenes orchids --expname gnt_llff --chunk_size 500 --run_val --N_samples 192

python3 eval.py --config configs/gnt_full.txt --eval_dataset rffr --eval_scenes art1 --expname gnt_full --chunk_size 500 --run_val --N_samples 192

# 直接利用Generalized NeRF的预训练权重在RFFR上进行finetune
export CUDA_VISIBLE_DEVICES=5
python3 eval.py --config configs/gnt_ft_rffr.txt --eval_dataset rffr --eval_scenes art1 --expname gen_ft_rffr --chunk_size 500 --run_val --N_rand 1024 --N_samples 192 --ckpt_path ./out/gnt_full/model_720000.pth



python3 render.py --config configs/gnt_full.txt --eval_scenes orchids --expname gnt_full --chunk_size 500 --run_val --N_samples 192

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train.py --config configs/gnt_ft_rffr.txt --expname vanilla_Nray --N_rand 2048


export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train.py --config configs/gnt_ft_rffr.txt --expname low_Nray --N_rand 1024

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train.py --config configs/gnt_ft_rffr.txt --expname vanilla_Nray --N_rand 1024 --N_samples 192


python -m torch.distributed.launch --nproc_per_node=8 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/cra/train_gnt_scannet.yaml