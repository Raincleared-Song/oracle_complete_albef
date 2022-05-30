# Oracle Completion (ALBEF)
The source code for graduation project *Completion of Oracle Bone Inscription Contents with Cross-Modal Model*. (Based on ALBEF)
### Environment
The environment is managed by anaconda3. Details are provided in the file `env.yaml`
### Training
Some training scripts are provided in `train.sh`.
#### First Step
Pretraining of ViT:
```bash
python3 Image_Reconstruct.py --config ./configs/Image_Reconstruct.yaml --mode both --save_all=true
```
To choose the best model:
```bash
python3 best_epoch.py -t tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd -m valid_accuracy_pre --check_only
```
Actually, We have already upload the models to this [link](https://cloud.tsinghua.edu.cn/d/f18e87629fbb4b598994/). The model chosen for the first stage is output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth
#### Second Step
Training of our oracle bone inscription completion model (using uploaded model):
```bash
python3 Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --text_encoder '' --mode both --load_cross \
  --image_checkpoint output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth
```
### Testing
To check the best model and its performance:
```bash
python3 best_epoch.py -t tra_finetune_single_mlm_p0_load_image_mk25_unrec_new --test_only
```
The best model is also uploaded to the same [link](https://cloud.tsinghua.edu.cn/d/f18e87629fbb4b598994/). The model chosen for the second stage is output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new/checkpoint_45.pth

To test other file, process it into the format like  `/data/private/songchenyang/hanzi_filter/handa/cases_com_tra_mid_combine.json`, and then use the following command:
```bash
python Finetune_single_mlm.py --config ./configs/Finetune_single_mlm.yaml --checkpoint output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new/checkpoint_45.pth --test_files {test_file} --do_trans false
```
And the result can be obtained under `output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new/logs_test`
