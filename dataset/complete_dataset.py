import os
import torch
import random
import numpy as np
from PIL import Image
from utils import load_json
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.data_utils import resize_pad_image


def process_complete(book: dict, config):
    grid_len = config['grid_len']
    img_res = config['image_res']
    pad_color = config['pad_color']
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
        pad_img = resize_pad_image(cur_img, (grid_res, grid_res), do_trans=config['img_random_transform'],
                                   pad_color=pad_color, mask_ratio=config['img_mask_ratio'],
                                   noise_ratio=config['img_noise_ratio'], do_rotate=config['img_do_rotate'])
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
            if token not in self.tokenizer.vocab:
                print('---', token, '---')
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


def process_single_complete(book: dict, config):
    img_res, img_mode, pad_color = config['image_res'], config['img_mode'], config['pad_color']
    res_images, res_caption = [], []
    for cid, (ch, image) in enumerate(book['characters']):
        assert os.path.exists(os.path.join(config['data_prefix'], image))
        cur_img = Image.open(os.path.join(config['data_prefix'], image)).convert(img_mode)
        res_caption.append(ch)
        pad_img = resize_pad_image(cur_img, (img_res, img_res), do_trans=False, pad_color=pad_color)
        pad_mask_img = resize_pad_image(cur_img, (img_res, img_res), do_trans=config['img_random_transform'],
                                        pad_color=pad_color, mask_ratio=config['img_mask_ratio'],
                                        noise_ratio=config['img_noise_ratio'], do_rotate=config['img_do_rotate'])
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
                    if config['modality'] == 'text' and len(book['characters']) == 1:
                        continue
                    for cid in range(len(book['characters'])):
                        self.data.append((book, cid))
            elif self.mode == 'char':
                for book in books:
                    for cid, (ch, img) in enumerate(book['characters']):
                        self.data.append({
                            'book_name': book['book_name'],
                            'row_order': book['row_order'],
                            'characters': [(ch, img)]
                        })
            elif self.mode == 'all_mask_char':
                for book in books:
                    for cid, (ch, img) in enumerate(book['characters']):
                        self.data.append(({
                            'book_name': book['book_name'],
                            'row_order': book['row_order'],
                            'characters': [(ch, img)]
                        }, 0))
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
        self.model_mode = mode
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
            if token not in self.tokenizer.vocab:
                print('---', token, '---')
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
        plain_targets = input_ids.clone()
        plain_targets[input_ids == self.tokenizer.pad_token_id] = -100
        plain_targets[input_ids == self.tokenizer.cls_token_id] = -100
        plain_targets[input_ids == self.tokenizer.sep_token_id] = -100

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return input_ids, targets, plain_targets, masked_indices.tolist()[1:-1], mask_id, mask_ch

    def mask_by_id(self, input_ids, mask_id):
        masked_indices = torch.full(input_ids.shape, False)
        masked_indices[mask_id] = True
        assert input_ids[mask_id].item() not in [self.tokenizer.pad_token_id,
                                                 self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        mask_ch = input_ids[mask_id].item()

        targets = input_ids.clone()
        targets[~masked_indices] = -100  # We only compute loss on masked tokens
        plain_targets = input_ids.clone()
        plain_targets[input_ids == self.tokenizer.pad_token_id] = -100
        plain_targets[input_ids == self.tokenizer.cls_token_id] = -100
        plain_targets[input_ids == self.tokenizer.sep_token_id] = -100

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return input_ids, targets, plain_targets, masked_indices.tolist()[1:-1], mask_id, mask_ch

    def get_input_images(self, image_ls, masked_indices):
        assert self.config['img_mask_policy'] in ['single', 'neighbour', 'all']
        assert len(image_ls) == len(masked_indices)
        assert np.sum(masked_indices) == 1  # temp
        masked_id = np.nonzero(masked_indices)[0][0]
        mask_ori_img = self.transform(image_ls[masked_id][0]).view(-1).unsqueeze(0)
        assert masked_indices[masked_id] == 1
        if self.config['img_mask_policy'] == 'neighbour':
            if masked_id > 0:
                masked_indices[masked_id - 1] = 1
            if masked_id < len(masked_indices) - 1:
                masked_indices[masked_id + 1] = 1
        elif self.config['img_mask_policy'] == 'all':
            masked_indices = [1] * len(masked_indices)
        images = torch.cat([self.transform(img[int(masked)]).view(-1).unsqueeze(0)
                            for img, masked in zip(image_ls, masked_indices)], dim=0)
        return masked_id, images, mask_ori_img

    def random_crop_characters(self, book, mid=-1):
        limit, chars, new_mid = self.config['max_length'], book['characters'], mid
        new_book = book.copy()
        if limit < 0:
            return new_book, mid
        if len(chars) > limit:
            if mid >= 0:
                begin = max(0, mid - limit // 2)
                new_mid = mid - begin
                assert 0 <= new_mid <= mid
            else:
                begin = random.randint(0, len(chars) - limit)
            new_book['characters'] = chars[begin:(begin+limit)]
        return new_book, new_mid

    def __getitem__(self, index):
        if self.mode in ['normal', 'char'] and not self.specific_test:
            book = self.data[index]
            book, _ = self.random_crop_characters(book)
            identity = book['book_name'] + '-' + str(book['row_order'])
            image_ls, tokens = process_single_complete(book, self.config)
            input_ids = torch.LongTensor(self.convert_tokens_to_ids(tokens))
            if self.add_mask:
                input_ids, targets, plain_targets, masked_indices, mask_id, mask_ch = self.random_mask(input_ids)
                masked_id, images, mask_ori_img = self.get_input_images(image_ls, masked_indices)
                return images, input_ids, targets, plain_targets, identity, mask_id, mask_ch, mask_ori_img
            else:
                images = torch.cat([self.transform(img[0]).view(-1).unsqueeze(0) for img in image_ls], dim=0)
                return images, input_ids, identity
        else:
            book, mid = self.data[index]
            assert mid < len(book['characters'])
            book, mid = self.random_crop_characters(book, mid)
            identity = book['book_name'] + '-' + str(book['row_order'])
            image_ls, tokens = process_single_complete(book, self.config)
            input_ids = torch.LongTensor(self.convert_tokens_to_ids(tokens))
            if self.add_mask:
                input_ids, targets, plain_targets, masked_indices, mask_id, mask_ch = self.mask_by_id(input_ids, mid+1)
                masked_id, images, mask_ori_img = self.get_input_images(image_ls, masked_indices)
                return images, input_ids, targets, plain_targets, identity, mask_id, mask_ch, mask_ori_img
            else:
                images = torch.cat([self.transform(img[0]).view(-1).unsqueeze(0) for img in image_ls], dim=0)
                return images, input_ids, identity
