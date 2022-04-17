import torch
import torch.nn as nn
from functools import partial
from transformers import RobertaConfig, RobertaForMaskedLM
from models.vit import interpolate_pos_embed, VisionTransformer, Block


class SingleMlm(nn.Module):
    """
    单个 BERT 的跨模态模型
    """
    def __init__(self, text_encoder=None, tokenizer=None, config=None, init_deit=True, distributed=False):
        super().__init__()

        self.tokenizer = tokenizer
        roberta_config = RobertaConfig.from_json_file(config['bert_config'])
        if text_encoder:
            self.text_encoder = RobertaForMaskedLM.from_pretrained(text_encoder, config=roberta_config)
        else:
            self.text_encoder = RobertaForMaskedLM(config=roberta_config)
        self.distributed = distributed
        self.modality = config['modality'] if 'modality' in config else 'cross'
        if 'topk' in config:
            assert config['mlm_probability'] <= 0
            self.topk = config['topk']

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

        if config['image_reconstruct_factor'] > 0:
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
        assert self.config['image_reconstruct_factor'] > 0
        embeds = embeds.unsqueeze(1)
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

    def forward(self, images, mask_ori_images, input_ids, attn_masks, labels, pos_ids, type_ids, lengths,
                mask_ids, mask_img_ids, mask_chs, mode):
        """
        input_ids: [batch_size, 1+n1+1+n2]
        images: [batch_size, n1+n2, res * res * chan]
        """
        assert mode in ['train', 'valid', 'test']

        if self.modality == 'cross':
            # word embedding
            word_embed = self.text_encoder.roberta.embeddings.word_embeddings(input_ids)  # (batch, 1+n1+1+n2, 768)
            visual_embed = self.forward_encoder(images)  # (batch, n1+n2, 768)
            assert word_embed.shape[1] == visual_embed.shape[1] + 2
            input_embeds = torch.cat((word_embed, visual_embed), dim=1)
        elif self.modality == 'text':
            input_embeds = self.text_encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            input_embeds = self.forward_encoder(images)

        mlm_output = self.text_encoder(
            input_ids=None,
            attention_mask=attn_masks,
            token_type_ids=type_ids,
            position_ids=pos_ids,
            inputs_embeds=input_embeds,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
        )

        loss_mlm, loss_rec = mlm_output.loss, torch.tensor(0.0).to(mlm_output.loss)

        batch_sz = len(lengths)
        if self.config['image_reconstruct_factor'] > 0:
            last_hidden = mlm_output.hidden_states[-1]
            img_embeds = last_hidden[torch.arange(batch_sz), mask_img_ids, :]
            loss_rec = self.forward_decoder(img_embeds, mask_ori_images)[1]

        with torch.no_grad():
            prediction_scores = mlm_output.logits
            # batch_size, sample_number, vocab_size
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

            # statistics for topk result
            batch_sz, sent_sz, vocab_sz = prediction_scores.shape
            prediction_scores = prediction_scores[torch.arange(batch_sz), mask_ids, :]
            # (bacth_sz, max_topk)
            topk_ids = torch.topk(prediction_scores, k=max(self.topk), dim=1)[1].tolist()
            rank_correct_num, rank_instance_num, hit_correct = {}, {}, {}
            assert len(topk_ids) == len(mask_chs) == batch_sz
            for k in self.topk:
                local_rank_correct, local_correct = 0, 0
                for lab, topk_id in zip(mask_chs, topk_ids):
                    local_top = topk_id[:k]
                    if lab not in local_top:
                        continue
                    local_correct += 1
                    local_rank_correct += k - local_top.index(lab)
                rank_correct_num[k] = local_rank_correct
                rank_instance_num[k] = batch_sz * k
                hit_correct[k] = local_correct

        total_loss = loss_mlm + loss_rec * self.config['image_reconstruct_factor']
        return total_loss, loss_mlm, loss_rec, correct_num, instance_num, ori_input_ids.tolist(), correct_chars, \
            wrong_chars, rank_correct_num, rank_instance_num, hit_correct, topk_ids
