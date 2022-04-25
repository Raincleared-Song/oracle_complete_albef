set -e

# python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain.yaml --mode train
# python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain_simple.yaml --mode train

python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain.yaml --mode test
python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain_simple.yaml --mode test

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true
srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true
python Image_Reconstruct.py --config ./configs/Image_Reconstruct.yaml --mode both --save_all=true
srun -G 1 -c 4 --mem 16g python3 Image_Reconstruct.py --config ./configs/Image_Reconstruct.yaml --mode both --save_all=true
python Image_Reconstruct.py --config ./configs/Image_Reconstruct.yaml --mode test --checkpoint output/tra_image_reconstruct_vit_all/checkpoint_29.pth
python Image_Reconstruct.py --config ./output/tra_image_reconstruct_vit_all_mk50_lr4/config.yaml --mode test --checkpoint output/tra_image_reconstruct_vit_all_mk50_lr4/checkpoint_37.pth

python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode train_valid
python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode test --checkpoint output/handa_book_sharpen_unet_base_inv_96/checkpoint_09.pth
python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode test --checkpoint output/handa_book_sharpen_unet_scale_inv_96/checkpoint_09.pth

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/tra_finetune_single_mlm_p0_vit_norec/checkpoint_57.pth --text_encoder '' --mode test
python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/finetune_single_mlm_np_mk50/checkpoint_48.pth --text_encoder '' --mode test
python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/finetune_single_mlm_np_mk75/checkpoint_59.pth --text_encoder '' --mode test

grep -E "accuracy\": 8[3-9]\.[6-9]" log_test.txt
