python3 eval.py --config configs/gnt_full.txt --eval_scenes orchids --expname gnt_full --chunk_size 500 --run_val --N_samples 192
python3 eval.py --config configs/gnt_llff.txt --eval_scenes orchids --expname gnt_llff --chunk_size 500 --run_val --N_samples 192

python3 eval.py --config configs/gnt_full.txt --eval_dataset rffr --eval_scenes art1 --expname gnt_full --chunk_size 500 --run_val --N_samples 192



python3 render.py --config configs/gnt_full.txt --eval_scenes orchids --expname gnt_full --chunk_size 500 --run_val --N_samples 192
