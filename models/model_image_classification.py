import torch
import torch.nn as nn
from functools import partial
import torchvision.models as vision_models
from models.vit import interpolate_pos_embed, VisionTransformer


class ImageClassification(nn.Module):
    """
    只做图像复原任务的单模态模型
    """
    def __init__(self, config=None, init_deit=True, distributed=False):
        super().__init__()

        self.distributed = distributed
        self.channel_num = len(config['img_mode'])
        input_number_classes = config['image_res'] * config['image_res'] * self.channel_num
        if config['visual_encoder'] == 'mlp':
            self.visual_encoder = nn.Sequential(
                nn.Linear(input_number_classes, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 768),
                nn.ReLU(),
            )
        elif config['visual_encoder'] == 'vit':
            self.visual_encoder = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=config['encoder_layer'], num_heads=12,
                in_chans=self.channel_num, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            if init_deit:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True)
                state_dict = checkpoint["model"]
                pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
                state_dict['pos_embed'] = pos_embed_reshaped
                msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
                print('ViT Initialization:', msg)
        elif config['visual_encoder'].startswith('resnet'):
            self.visual_encoder = {
                'resnet18':  vision_models.resnet18,
                'resnet50':  vision_models.resnet50,
                'resnet101': vision_models.resnet101,
                'resnet152': vision_models.resnet152,
            }[config['visual_encoder']](pretrained=True)
            dim_mlp = self.visual_encoder.fc.in_features
            self.visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 768))
            print(f'{config["visual_encoder"]} Initialization ......')
        else:
            raise ValueError('Invalid Visual Encoder!')

        self.classification_head = nn.Linear(768, config['vocab_size'])
        self.classification_loss = nn.CrossEntropyLoss()
        self.config = config

    def forward_encoder(self, images):
        if self.config['visual_encoder'] == 'vit':
            image_embeds = self.visual_encoder(images)[:, 0, :]
        else:
            image_embeds = self.visual_encoder(images)
        return image_embeds

    def forward_classification(self, embeds, labels):
        instance_num, = labels.shape
        assert torch.sum(labels == -100) == 0
        predict_embeds = self.classification_head(embeds)
        loss_cls = self.classification_loss(predict_embeds, labels)
        with torch.no_grad():
            predict_result_ids = torch.max(predict_embeds, dim=1)[1]
            predict_result = predict_result_ids == labels
            correct_num = torch.sum(torch.logical_and(labels, predict_result)).item()
        assert correct_num <= instance_num
        return loss_cls, predict_result, correct_num, instance_num

    def forward(self, images, labels, mode):
        assert mode in ['train', 'valid', 'test']

        input_embeds = self.forward_encoder(images)
        loss_cls, predict_result, correct_num, instance_num = self.forward_classification(input_embeds, labels)

        return loss_cls, predict_result, correct_num, instance_num
