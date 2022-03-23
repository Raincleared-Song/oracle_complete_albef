"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import random
from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xroberta import RobertaConfig, RobertaForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn


class AlbefSimple(nn.Module):
    """
    ALBEF model without momentum technique
    mask images during training time
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
        embed_dim = config['embed_dim']

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

        vision_width = config['vision_width']
        roberta_config = RobertaConfig.from_json_file(config['bert_config'])

        self.text_encoder = RobertaForMaskedLM.from_pretrained(text_encoder, config=roberta_config)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.itm_head = nn.Linear(text_width, 2)
        self.distributed = distributed

    def forward(self, images, input_ids, attn_masks, mode, alpha=0):
        assert mode in ['train', 'test']
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder.roberta(input_ids, attention_mask=attn_masks,
                                                return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        sim_i2t = image_feat @ text_feat.t() / self.temp
        sim_t2i = text_feat @ image_feat.t() / self.temp

        with torch.no_grad():
            sim_targets = torch.zeros(sim_i2t.size()).to(images.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i, dim=1) + (1 - alpha) * sim_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        # ================================= #
        # forward the positve image-text pair
        output_pos = self.text_encoder.roberta(encoder_embeds=text_embeds,
                                               attention_mask=attn_masks,
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts,
                                               return_dict=True,
                                               mode='fusion')
        with torch.no_grad():
            bs = images.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            except RuntimeError:
                width, height = weights_t2i.shape
                assert width == bs and torch.sum(weights_t2i[b]) == 0
                weights_t2i[b, :] = 1.0 / (height - 1)
                weights_t2i[b, b] = 0
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                # print('-----------------------------', end='\n\n\n')
                # from IPython import embed
                # embed()
                # raise err
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            except RuntimeError:
                width, height = weights_i2t.shape
                assert width == bs and torch.sum(weights_i2t[b]) == 0
                weights_i2t[b, :] = 1.0 / (height - 1)
                weights_i2t[b, b] = 0
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                # print('-----------------------------', end='\n\n\n')
                # from IPython import embed
                # embed()
                # raise err
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(attn_masks[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([attn_masks, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder.roberta(encoder_embeds=text_embeds_all,
                                               attention_mask=text_atts_all,
                                               encoder_hidden_states=image_embeds_all,
                                               encoder_attention_mask=image_atts_all,
                                               return_dict=True,
                                               mode='fusion')

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(images.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        # ================= MLM ======================== #
        mlm_input_ids = input_ids.clone()
        labels = mlm_input_ids.clone()

        mlm_input_ids, labels = self.mask(mlm_input_ids, self.text_encoder.config.vocab_size,
                                          images.device, targets=labels)

        mlm_output = self.text_encoder(mlm_input_ids,
                                       attention_mask=attn_masks,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       )
        loss_mlm = mlm_output.loss

        return loss_mlm, loss_ita, loss_itm

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None):
        if masked_indices is not None:
            pass
        elif self.mlm_probability > 0:
            probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            masked_indices[input_ids == self.tokenizer.pad_token_id] = False
            masked_indices[input_ids == self.tokenizer.cls_token_id] = False
            masked_indices[input_ids == self.tokenizer.sep_token_id] = False

            if torch.sum(masked_indices) == 0:
                candidates = []
                batch_sz, seq_len = input_ids.shape
                for bid in range(batch_sz):
                    for tid in range(seq_len):
                        if input_ids[bid, tid] not in (
                                self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id):
                            candidates.append((bid, tid))
                assert len(candidates) > 0
                masked_indices[random.choice(candidates)] = True
        else:
            mask_char_num = - int(self.mlm_probability)
            masked_indices = torch.full(input_ids.shape, False)
            assert mask_char_num == 1  # temp
            batch_sz, seq_len = input_ids.shape
            for bid in range(batch_sz):
                candidates = []
                for tid in range(seq_len):
                    if input_ids[bid, tid] not in (
                            self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id):
                        candidates.append(tid)
                assert len(candidates) > 0
                masked_indices[bid, random.choice(candidates)] = True
            assert torch.sum(masked_indices) == mask_char_num * batch_sz

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
