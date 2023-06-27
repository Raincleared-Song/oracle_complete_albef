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
python Image_Reconstruct.py --config ./output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd_new/config.yaml --mode test --checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd_new/checkpoint_89.pth
python Image_Reconstruct.py --config ./output/tra_image_reconstruct_vit_all_mk25_ranoi_rot_cls_lr4_upd/config.yaml --mode test --checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_rot_cls_lr4_upd/checkpoint_43.pth
python Image_Reconstruct.py --config ./output/tra_image_reconstruct_vit_all_mk25_ranoi_rot_lr4_upd/config.yaml --mode test --checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_rot_lr4_upd/checkpoint_56.pth

python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode train_valid
python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode test --checkpoint output/handa_book_sharpen_unet_base_inv_96/checkpoint_09.pth
python Sharpen_unet.py --config ./configs/Sharpen_unet.yaml --mode test --checkpoint output/handa_book_sharpen_unet_scale_inv_96/checkpoint_09.pth

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/tra_finetune_single_mlm_p0_text_all/checkpoint_15.pth --text_encoder '' --mode test
python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/finetune_single_mlm_np_mk50/checkpoint_48.pth --text_encoder '' --mode test

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/tra_finetune_single_mlm_p0_load_image_mk25_unrec/checkpoint_45.pth --mode test

grep -E "accuracy\": 8[3-9]\.[6-9]" log_test.txt

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --text_checkpoint output/tra_finetune_single_mlm_p0_text_all/checkpoint_15.pth \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth

srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth
srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth
srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '../chinese-bert-wwm-ext' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth

srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --text_checkpoint output/tra_finetune_single_mlm_p0_text_all/checkpoint_15.pth \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth

# final best result for tiny manual set
python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode test \
  --checkpoint output/tra_finetune_single_mlm_p0_load_cross01/checkpoint_44.pth

python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode test \
  --checkpoint output/tra_finetune_single_mlm_p0_load_cross02/checkpoint_53.pth

# no cls pretrain
python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --text_checkpoint output/tra_finetune_single_mlm_p0_text_all/checkpoint_15.pth \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_lr4_upd/checkpoint_52.pth

srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_lr4_upd/checkpoint_52.pth

srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_cls_lr4_upd/checkpoint_44.pth

srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk50_ranoi_cls_lr4_fac10_upd/checkpoint_43.pth

# mk75
srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk75_ranoi_cls_lr4_fac10_upd/checkpoint_28.pth

# only OIC
srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_ori_cls_lr4_upd/checkpoint_50.pth

# - RIC
srun -G 1 -c 4 --mem 16g python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --save_all=true --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ori_cls_lr4_upd/checkpoint_55.pth

srun -G 1 -c 4 --mem 16g python3 Image_Classification.py --config ./configs/Image_Classification.yaml --mode both --save_all=true
srun -G 1 -c 4 --mem 16g python3 Image_Classification.py --config ./configs/Image_Classification.yaml --mode both --save_all=true \
  --pretrained_encoder output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth
srun -G 1 -c 4 --mem 16g python3 Image_Classification.py --config ./configs/Image_Classification.yaml --mode both --save_all=true \
  --pretrained_encoder output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new/checkpoint_45.pth
srun -G 1 -c 4 --mem 16g python3 Image_Classification.py --config ./configs/Image_Classification.yaml --mode both --save_all=true \
  --pretrained_encoder output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_wwm_all_noinit_mlm2/checkpoint_49.pth

python3 Image_Classification.py --config ./output/image_class_chant_vit_pre_second89/config.yaml --mode test \
  --checkpoint output/image_class_chant_vit_pre_second89/checkpoint_02.pth \
  --test_files orcal/oracle_classification_chant_test.json --do_trans false

python Finetune_single_mlm.py --config output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new/config.yaml \
  --checkpoint output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new/checkpoint_45.pth --mode test \
  --test_files handa/cases_H00137zheng_see.json --do_trans false
python Finetune_single_mlm.py --config output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_wwm_all_noinit_mlm_rec10_b4_258/config.yaml \
  --checkpoint output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_wwm_all_noinit_mlm_rec10_b4_258/checkpoint_44.pth --mode test \
  --test_files handa/cases_H00137zheng_see.json --do_trans false --device cuda:0

python3 Image_Classification.py --config ./output/image_class_chant_vit_pre_second89/config.yaml --mode test \
  --checkpoint output/image_class_chant_vit_pre_second89/checkpoint_02.pth \
  --test_files handa/H32384_classification.json --do_trans false
python Finetune_single_mlm.py --config output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_wwm_all_noinit_mlm_rec10_b4_89/config.yaml \
  --checkpoint output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_wwm_all_noinit_mlm_rec10_b4_89/checkpoint_53.pth --mode test \
  --test_files handa/H32384_complete.json --do_trans false
