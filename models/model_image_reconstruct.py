import torch
import torch.nn as nn
from functools import partial
from models.vit import interpolate_pos_embed, VisionTransformer, Block


class ImageReconstruct(nn.Module):
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
        else:
            raise ValueError('Invalid Visual Encoder!')

        self.reconstruct_decoder = nn.ModuleList([
            Block(
                dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(config['decoder_layer'])
        ])
        self.reconstruct_norm = nn.LayerNorm(768)
        self.reconstruct_head = nn.Linear(768, input_number_classes)
        self.reconstruct_loss = nn.MSELoss()

        self.config = config
        # self.rec_idx = 0

    def forward_encoder(self, images):
        if self.config['visual_encoder'] == 'vit':
            batch_size, img_seq_len, img_pix_num = images.shape
            img_res, img_chan = self.config['image_res'], self.channel_num
            images = images.view(batch_size * img_seq_len, img_chan, img_res, img_res)
            vit_embeds = self.visual_encoder(images)[:, 0, :]
            image_embeds = vit_embeds.view(batch_size, img_seq_len, 768)
        else:
            image_embeds = self.visual_encoder(images)
        return image_embeds

    def forward_decoder(self, embeds, targets):
        for blk in self.reconstruct_decoder:
            embeds = blk(embeds)
        embeds = embeds.squeeze(1)
        embeds = self.reconstruct_head(self.reconstruct_norm(embeds))
        loss = self.reconstruct_loss(embeds, targets)
        # torch.save([embeds.cpu(), targets.cpu()], f'{self.config["output_path"]}/{self.rec_idx}.pth')
        # self.rec_idx += 1
        # if self.rec_idx == 10:
        #     exit()
        return embeds, loss

    def forward(self, images, labels, mode):
        """
        input_ids: [batch_size, 1+n1+1+n2]
        images: [batch_size, n1+n2, res * res * chan]
        """
        assert mode in ['train', 'valid', 'test']

        input_embeds = self.forward_encoder(images)
        loss_rec = self.forward_decoder(input_embeds, labels)[1]

        return loss_rec
