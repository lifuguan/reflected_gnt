#!/bin/bash

# Function to execute a command and check for errors
execute_command() {
    echo "Executing: $1"
    eval $1
    if [ $? -ne 0 ]; then
        echo "Error executing the command: $1"
        exit 1
    fi
}

# Set up environment variables if needed

# Execute commands sequentially
execute_command "python -m torch.distributed.launch --nproc_per_node=4 --master_port=$(( RANDOM % 1000 + 50000 )) ft_scannet.py --config configs/gnt_replica.txt --ckpt_path ./out/gnt_best.pth --expname ft_replica_office2_no_eval --no_load_opt --no_load_scheduler --val_set_list configs/replica_office2.txt"
execute_command "python -m torch.distributed.launch --nproc_per_node=4 --master_port=$(( RANDOM % 1000 + 50000 )) ft_scannet.py --config configs/gnt_replica.txt --ckpt_path ./out/gnt_best.pth --expname ft_replica_office3_no_eval --no_load_opt --no_load_scheduler --val_set_list configs/replica_office3.txt"
execute_command "python -m torch.distributed.launch --nproc_per_node=4 --master_port=$(( RANDOM % 1000 + 50000 )) ft_scannet.py --config configs/gnt_replica.txt --ckpt_path ./out/gnt_best.pth --expname ft_replica_office4_no_eval --no_load_opt --no_load_scheduler --val_set_list configs/replica_office4.txt"
execute_command "python -m torch.distributed.launch --nproc_per_node=4 --master_port=$(( RANDOM % 1000 + 50000 )) ft_scannet.py --config configs/gnt_replica.txt --ckpt_path ./out/gnt_best.pth --expname ft_replica_room0_no_eval --no_load_opt --no_load_scheduler --val_set_list configs/replica_room0.txt"
execute_command "python -m torch.distributed.launch --nproc_per_node=4 --master_port=$(( RANDOM % 1000 + 50000 )) ft_scannet.py --config configs/gnt_replica.txt --ckpt_path ./out/gnt_best.pth --expname ft_replica_room1_no_eval --no_load_opt --no_load_scheduler --val_set_list configs/replica_room1.txt"
execute_command "python -m torch.distributed.launch --nproc_per_node=4 --master_port=$(( RANDOM % 1000 + 50000 )) ft_scannet.py --config configs/gnt_replica.txt --ckpt_path ./out/gnt_best.pth --expname ft_replica_room2_no_eval --no_load_opt --no_load_scheduler --val_set_list configs/replica_room3.txt"
