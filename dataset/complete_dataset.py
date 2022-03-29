import os
import torch
import random
import numpy as np
from PIL import Image
from utils import load_json
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.data_utils import resize_pad_image


def process_complete(book: dict, config, pad_color=255):
    grid_len = config['grid_len']
    img_res = config['image_res']
    assert img_res % grid_len == 0
    grid_res = img_res // grid_len
    chars = book['characters']
    max_img_num = grid_len * grid_len
    if len(chars) > max_img_num:
        # random sample
        start = random.randint(0, len(chars)-max_img_num)
        chars = chars[start:start+max_img_num]
    res_img, res_caption = np.full((img_res, img_res, 3), pad_color, dtype=np.uint8), []
    for cid, (ch, image) in enumerate(chars):
        assert os.path.exists(os.path.join(config['data_prefix'], image))
        cur_img = Image.open(os.path.join(config['data_prefix'], image)).convert('RGB')
        res_caption.append(ch)
        h_idx, w_idx = cid // grid_len, cid % grid_len
        pad_img = resize_pad_image(cur_img, (grid_res, grid_res), pad_color=pad_color)
        res_img[h_idx*grid_res:(h_idx+1)*grid_res, w_idx*grid_res:(w_idx+1)*grid_res, :] = pad_img
    res_img = Image.fromarray(res_img, mode='RGB')
    return res_img, res_caption


class OracleCompleteDataset(Dataset):
    def __init__(self, config, mode, tokenizer, add_mask=False):
        self.data = []
        file_list = config['train_file'] if mode == 'train' else config['test_file']
        for file in file_list:
            self.data += load_json(os.path.join(config['data_prefix'], file))
        # self.data = self.data[:100]
        self.config = config
        mean, std = {
            512: ([0.8441, 0.8440, 0.8439], [0.3207, 0.3208, 0.3210]),
            368: ([0.8444, 0.8443, 0.8442], [0.3201, 0.3202, 0.3203]),
            320: ([0.8446, 0.8445, 0.8444], [0.3196, 0.3197, 0.3199]),
        }[config['image_res']]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.add_mask = add_mask
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def convert_tokens_to_ids(self, tokens):
        for token in tokens:
            assert token in self.tokenizer.vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        return input_ids

    def mask(self, input_ids):
        if self.config['mlm_probability'] > 0:
            probability_matrix = torch.full(input_ids.shape, self.config['mlm_probability'])
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
            mask_char_num = - int(self.config['mlm_probability'])
            masked_indices = torch.full(input_ids.shape, False)
            assert mask_char_num == 1  # temp
            seq_len, = input_ids.shape
            candidates = []
            for tid in range(seq_len):
                if input_ids[tid] not in (
                        self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id):
                    candidates.append(tid)
            assert len(candidates) > 0
            masked_indices[random.choice(candidates)] = True
            assert torch.sum(masked_indices) == mask_char_num

        targets = input_ids.clone()
        targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return input_ids, targets

    def __getitem__(self, index):
        img, tokens = process_complete(self.data[index], self.config)
        img = self.transform(img)
        input_ids = torch.LongTensor(self.convert_tokens_to_ids(tokens))
        if self.add_mask:
            input_ids, targets = self.mask(input_ids)
            return img, input_ids, targets
        else:
            return img, input_ids


def process_single_complete(book: dict, config, pad_color=255):
    img_res, img_mode = config['image_res'], config['img_mode']
    res_images, res_caption = [], []
    for cid, (ch, image) in enumerate(book['characters']):
        assert os.path.exists(os.path.join(config['data_prefix'], image))
        cur_img = Image.open(os.path.join(config['data_prefix'], image)).convert(img_mode)
        res_caption.append(ch)
        pad_img = resize_pad_image(cur_img, (img_res, img_res),
                                   pad_color=pad_color, mask_ratio=0.0)
        pad_mask_img = resize_pad_image(cur_img, (img_res, img_res),
                                        pad_color=pad_color, mask_ratio=config['img_mask_ratio'])
        res_images.append((Image.fromarray(pad_img, mode=img_mode), Image.fromarray(pad_mask_img, mode=img_mode)))
    return res_images, res_caption


class OracleCompleteSingleDataset(Dataset):
    def __init__(self, config, mode, tokenizer, add_mask=False):
        self.mode = config['dataset_mode']
        self.data = []
        file_list = config['train_file'] if mode == 'train' else config['test_file']
        self.specific_test = config['specific_test'] and mode != 'train'
        for file in file_list:
            books = load_json(os.path.join(config['data_prefix'], file))
            if self.specific_test:
                assert isinstance(books[0], list)
                for book, cid in books:
                    if self.mode != 'char':
                        self.data.append((book, cid))
                    else:
                        self.data.append(({'book_name': book['book_name'],
                                           'row_order': book['row_order'],
                                           'characters': [book['characters'][cid]]}, 0))
                continue
            # 提前检查是否存在空缺字符
            if self.mode != 'complete':
                for book in books:
                    for ch, _ in book['characters']:
                        assert ch != '■'
            if self.mode == 'normal':
                self.data += books
            elif self.mode == 'all_mask':
                for book in books:
                    for cid in range(len(book['characters'])):
                        self.data.append((book, cid))
            elif self.mode == 'char':
                for book in books:
                    for cid, (ch, img) in enumerate(book['characters']):
                        self.data.append({'characters': [(ch, img)]})
            elif self.mode == 'all_mask_char':
                for book in books:
                    for cid, (ch, img) in enumerate(book['characters']):
                        self.data.append(({'characters': [(ch, img)]}, 0))
                        if len(book['characters']) > 1:
                            self.data.append((book, cid))
            elif self.mode == 'complete':
                for book in books:
                    for cid, (ch, img) in enumerate(book['characters']):
                        if ch == '■':
                            self.data.append((book, cid))
            else:
                raise ValueError('config dataset_mode')
        self.config = config
        if config['img_mode'] == 'RGB':
            mean, std = {
                128: ([0.5601, 0.5598, 0.5596], [0.4064, 0.4065, 0.4066]),
            }[config['image_res']]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            mean, std = {
                128: ([0.5599], [0.4065]),
            }[config['image_res']]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        self.add_mask = add_mask
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def convert_tokens_to_ids(self, tokens):
        for token in tokens:
            assert token in self.tokenizer.vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        return input_ids

    def random_mask(self, input_ids):
        if self.config['mlm_probability'] > 0:
            probability_matrix = torch.full(input_ids.shape, self.config['mlm_probability'])
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
                                self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                        ):
                            candidates.append((bid, tid))
                assert len(candidates) > 0
                masked_indices[random.choice(candidates)] = True
            mask_id, mask_ch = -1, -1
        else:
            mask_char_num = - int(self.config['mlm_probability'])
            masked_indices = torch.full(input_ids.shape, False)
            assert mask_char_num == 1  # temp
            seq_len, = input_ids.shape
            candidates = []
            for tid in range(seq_len):
                if input_ids[tid] not in (
                        self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                ):
                    candidates.append(tid)
            assert len(candidates) > 0
            mask_id = random.choice(candidates)
            mask_ch = input_ids[mask_id].item()
            masked_indices[mask_id] = True
            assert torch.sum(masked_indices) == mask_char_num

        targets = input_ids.clone()
        targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return input_ids, targets, masked_indices.tolist()[1:-1], mask_id, mask_ch

    def mask_by_id(self, input_ids, mask_id):
        masked_indices = torch.full(input_ids.shape, False)
        masked_indices[mask_id] = True
        assert input_ids[mask_id].item() not in [self.tokenizer.pad_token_id,
                                                 self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        mask_ch = input_ids[mask_id].item()

        targets = input_ids.clone()
        targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return input_ids, targets, masked_indices.tolist()[1:-1], mask_id, mask_ch

    def __getitem__(self, index):
        if self.mode in ['normal', 'char'] and not self.specific_test:
            book = self.data[index]
            identity = book['book_name'] + '-' + str(book['row_order'])
            images, tokens = process_single_complete(self.data[index], self.config)
            input_ids = torch.LongTensor(self.convert_tokens_to_ids(tokens))
            if self.add_mask:
                input_ids, targets, masked_indices, mask_id, mask_ch = self.random_mask(input_ids)
                assert len(images) == len(masked_indices)
                images = torch.cat([self.transform(img[int(masked)]).view(-1).unsqueeze(0)
                                    for img, masked in zip(images, masked_indices)], dim=0)
                return images, input_ids, targets, identity, mask_id, mask_ch
            else:
                images = torch.cat([self.transform(img[0]).view(-1).unsqueeze(0) for img in images], dim=0)
                return images, input_ids, identity
        else:
            book, mid = self.data[index]
            identity = book['book_name'] + '-' + str(book['row_order'])
            images, tokens = process_single_complete(book, self.config)
            input_ids = torch.LongTensor(self.convert_tokens_to_ids(tokens))
            if self.add_mask:
                input_ids, targets, masked_indices, mask_id, mask_ch = self.mask_by_id(input_ids, mid + 1)
                assert len(images) == len(masked_indices)
                images = torch.cat([self.transform(img[int(masked)]).view(-1).unsqueeze(0)
                                    for img, masked in zip(images, masked_indices)], dim=0)
                return images, input_ids, targets, identity, mask_id, mask_ch
            else:
                images = torch.cat([self.transform(img[0]).view(-1).unsqueeze(0) for img in images], dim=0)
                return images, input_ids, identity
