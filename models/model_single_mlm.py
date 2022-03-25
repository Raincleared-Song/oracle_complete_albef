import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaForMaskedLM


class SingleMlm(nn.Module):
    """
    单个 BERT 的跨模态模型
    """
    def __init__(self, text_encoder=None, tokenizer=None, config=None, distributed=False):
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

        self.visual_encoder = nn.Sequential(
            nn.Linear(config['image_res'] * config['image_res'] * len(config['img_mode']), 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
        )

    def forward(self, images, input_ids, attn_masks, labels, pos_ids, type_ids, lengths, mask_ids, mask_chs, mode):
        """
        input_ids: [batch_size, 1+n1+1+n2]
        images: [batch_size, n1+n2, res * res * chan]
        """
        assert mode in ['train', 'test']

        if self.modality == 'cross':
            # word embedding
            word_embed = self.text_encoder.roberta.embeddings.word_embeddings(input_ids)  # (batch, 1+n1+1+n2, 768)
            visual_embed = self.visual_encoder(images)  # (batch, n1+n2, 768)
            assert word_embed.shape[1] == visual_embed.shape[1] + 2
            input_embeds = torch.cat((word_embed, visual_embed), dim=1)
        elif self.modality == 'text':
            input_embeds = self.text_encoder.roberta.embeddings.word_embeddings(input_ids)
        else:
            input_embeds = self.visual_encoder(images)

        mlm_output = self.text_encoder(
            input_ids=None,
            attention_mask=attn_masks,
            token_type_ids=type_ids,
            position_ids=pos_ids,
            inputs_embeds=input_embeds,
            labels=labels,
            return_dict=True,
        )

        loss_mlm = mlm_output.loss

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

        return loss_mlm, correct_num, instance_num, ori_input_ids.tolist(), correct_chars, wrong_chars, \
            rank_correct_num, rank_instance_num, hit_correct, topk_ids
