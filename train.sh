set -e

# python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain.yaml --mode train
# python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain_simple.yaml --mode train

python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain.yaml --mode test
python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain_simple.yaml --mode test

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true
