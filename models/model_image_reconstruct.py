import torch
import torch.nn as nn
from functools import partial
from models.vit import interpolate_pos_embed, VisionTransformer, Block


class ImageReconstruct(nn.Module):
    """
    只做图像复原任务的单模态模型
    """
    def __init__(self, config=None, tokenizer=None, init_deit=True, distributed=False):
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
        self.reconstruct_head = nn.Linear(768, 768)
        self.reconstruct_conv = nn.ConvTranspose2d(768, self.channel_num, kernel_size=16, stride=16)
        self.reconstruct_loss = nn.MSELoss()

        self.config = config
        self.tokenizer = tokenizer
        # self.rec_idx = 0

        self.image_classification_factor = config['image_classification_factor']
        if sum(self.image_classification_factor) > 0:
            self.classification_head = nn.Linear(768, self.tokenizer.vocab_size)
            self.classification_loss = nn.CrossEntropyLoss()

    def forward_encoder(self, images):
        batch_size, img_seq_len, img_pix_num = images.shape
        if self.config['visual_encoder'] == 'vit':
            img_res, img_chan = self.config['image_res'], self.channel_num
            images = images.view(batch_size * img_seq_len, img_chan, img_res, img_res)
            patch_embeds = self.visual_encoder(images)  # [batch_size * img_seq_len, patch_num, hidden_size]
            cls_embeds = patch_embeds[:, 0, :]
            cls_embeds = cls_embeds.view(batch_size, img_seq_len, 768)
        else:
            cls_embeds = self.visual_encoder(images)  # [batch_size, img_seq_len, hidden_size]
            patch_embeds = cls_embeds.view(batch_size * img_seq_len, 1, 768)
        return cls_embeds, patch_embeds

    def forward_decoder(self, embeds, targets, images=None):
        for blk in self.reconstruct_decoder:
            embeds = blk(embeds)
        batch_size, seq_len, pix_num = targets.shape
        cls_embeds, patch_embeds = embeds[:, 0, :].view(batch_size, seq_len, 768), embeds[:, 1:, :]
        img_pre = self.reconstruct_head(self.reconstruct_norm(patch_embeds))
        patch_num_edge = self.config['image_res'] // 16
        img_pre = img_pre.transpose(1, 2).view(batch_size * seq_len, 768, patch_num_edge, patch_num_edge)
        img_pre = self.reconstruct_conv(img_pre).reshape(batch_size, seq_len, pix_num)
        loss = self.reconstruct_loss(img_pre, targets)
        # if images is None:
        #     torch.save([img_pre.cpu(), targets.cpu()], f'{self.config["output_path"]}/{self.rec_idx}.pth')
        # else:
        #     torch.save([img_pre.cpu(), targets.cpu(), images.cpu()],
        #                f'{self.config["output_path"]}/{self.rec_idx}.pth')
        # self.rec_idx += 1
        # if self.rec_idx == 10:
        #     exit()
        return img_pre, cls_embeds, loss

    def forward_classification(self, embeds, texts):
        batch_sz, seq_len = texts.shape
        label_mask = texts != -100
        instance_num = torch.sum(label_mask).item()
        if sum(self.image_classification_factor) > 0:
            txt_embeds = self.classification_head(embeds)
            loss_cls = self.classification_loss(txt_embeds.view(batch_sz * seq_len, -1), texts.view(-1))
            with torch.no_grad():
                predict_result_ids = torch.max(txt_embeds, dim=2)[1]
                correct_num = torch.sum(torch.logical_and(label_mask, predict_result_ids == texts)).item()
        else:
            loss_cls, correct_num = torch.tensor(0.0).to(embeds), 0
        assert correct_num <= instance_num
        return loss_cls, correct_num, instance_num

    def forward(self, images, labels, texts, mode):
        """
        input_ids: [batch_size, 1+n1+1+n2]
        images: [batch_size, n1+n2, res * res * chan]
        """
        assert mode in ['train', 'valid', 'test']

        input_embeds, input_patch_embeds = self.forward_encoder(images)
        _, pre_embeds, loss_rec = self.forward_decoder(input_patch_embeds, labels, images)
        # predict the character by embeds predicted
        loss_cls_pre, correct_pre, instance_pre = self.forward_classification(pre_embeds, texts)
        tar_embeds, _ = self.forward_encoder(labels)
        # predict the character by embeds targeted
        loss_cls_ori, correct_ori, instance_ori = self.forward_classification(tar_embeds, texts)
        assert instance_pre == instance_ori

        total_loss = loss_rec * self.config['image_reconstruct_factor'] + \
            loss_cls_ori * self.image_classification_factor[0] + loss_cls_pre * self.image_classification_factor[1]

        return total_loss, loss_rec, loss_cls_ori, loss_cls_pre, correct_ori, correct_pre, instance_ori
