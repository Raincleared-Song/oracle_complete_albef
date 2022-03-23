"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xroberta import RobertaConfig, RobertaForMaskedLM

import torch
from torch import nn


class AlbefMlm(nn.Module):
    """
    ALBEF model without momentum technique for MLM
    mask images during preprocessing time
    """
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 init_deit=True,
                 distributed=False,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print('ViT Initialization:', msg)

        roberta_config = RobertaConfig.from_json_file(config['bert_config'])

        self.text_encoder = RobertaForMaskedLM.from_pretrained(text_encoder, config=roberta_config)
        self.distributed = distributed

    def forward(self, images, input_ids, attn_masks, labels, mode):
        assert mode in ['train', 'test']

        image_embeds = self.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)

        mlm_output = self.text_encoder(input_ids=input_ids,
                                       attention_mask=attn_masks,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       )
        loss_mlm = mlm_output.loss

        with torch.no_grad():
            prediction_scores = mlm_output.logits
            predict_result_ids = torch.argmax(prediction_scores, dim=2)
            label_mask = labels != -100
            predict_result = torch.logical_and(label_mask, predict_result_ids == labels)
            instance_num = torch.sum(label_mask)
            correct_num = torch.sum(predict_result)

            correct_ids = [tuple(idx) for idx in torch.nonzero(predict_result).cpu().tolist()]
            correct_chars = []
            for idx in correct_ids:
                correct_chars.append((labels[idx].item(), idx))

            ori_input_ids = input_ids.cpu()
            instance_idx = [tuple(idx) for idx in torch.nonzero(labels + 100).cpu().tolist()]
            wrong_chars = []
            correct_set = set(correct_ids)
            for idx in instance_idx:
                ori_input_ids[idx] = labels[idx].item()
                if idx not in correct_set:
                    wrong_chars.append((labels[idx].item(), predict_result_ids[idx].item(), idx))

        return loss_mlm, correct_num, instance_num, ori_input_ids.tolist(), correct_chars, wrong_chars
