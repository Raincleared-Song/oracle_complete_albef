import os
import torch
import random
from PIL import Image
from utils import load_json
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.data_utils import resize_pad_image


def process_image_reconstruct(book: dict, mids: list, config):
    img_res, img_mode, pad_color = config['image_res'], config['img_mode'], config['pad_color']
    res_images = []
    for cid, (_, image) in enumerate(book['characters']):
        if cid not in mids:
            continue
        assert os.path.exists(os.path.join(config['data_prefix'], image))
        cur_img = Image.open(os.path.join(config['data_prefix'], image)).convert(img_mode)
        pad_img = resize_pad_image(cur_img, (img_res, img_res), do_trans=False, pad_color=pad_color)
        pad_mask_img = resize_pad_image(cur_img, (img_res, img_res), do_trans=config['img_random_transform'],
                                        pad_color=pad_color, mask_ratio=config['img_mask_ratio'],
                                        noise_ratio=config['img_noise_ratio'], do_rotate=config['img_do_rotate'])
        res_images.append((Image.fromarray(pad_img, mode=img_mode), Image.fromarray(pad_mask_img, mode=img_mode)))
    assert len(res_images) > 0
    return res_images


def is_valid_image(book, cid):
    book_name, row_order = book['book_name'], book['row_order']
    return os.path.basename(book['characters'][cid][1]).startswith(f'{book_name}-{row_order}')


class ImageReconstructDataset(Dataset):
    def __init__(self, config, mode):
        self.mode, self.data, self.config = config['dataset_mode'], [], config
        file_list = config['train_file'] if mode == 'train' else config['test_file']
        self.specific_test = config['specific_test'] and mode != 'train'
        for file in file_list:
            books = load_json(os.path.join(config['data_prefix'], file))
            if self.specific_test:
                assert isinstance(books[0], list)
                for book, cid in books:
                    self.data.append((book, [cid]))
                continue
            if self.mode == 'normal':
                # 每个条目随机选 1 张图片
                for book in books:
                    book = self.random_crop_characters(book)
                    candidates = [cid for cid in range(len(book['characters'])) if is_valid_image(book, cid)]
                    if len(candidates) == 0:
                        continue
                    mid = random.choice(candidates)
                    self.data.append((book, [mid]))
            elif self.mode == 'all_mask':
                # 每个条目取所有图片
                for book in books:
                    book = self.random_crop_characters(book)
                    candidates = [cid for cid in range(len(book['characters'])) if is_valid_image(book, cid)]
                    if len(candidates) == 0:
                        continue
                    self.data.append((book, candidates))
            else:
                raise ValueError('config dataset_mode')
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

    def __len__(self):
        return len(self.data)

    def random_crop_characters(self, book):
        limit, chars = self.config['max_length'], book['characters']
        if limit < 0:
            return book
        if len(chars) > limit:
            begin = random.randint(0, len(chars) - limit)
            book['characters'] = chars[begin:(begin+limit)]
        return book

    def __getitem__(self, index):
        book, mids = self.data[index]
        image_ls = process_image_reconstruct(book, mids, self.config)
        images = torch.cat([self.transform(img[1]).view(-1).unsqueeze(0) for img in image_ls], dim=0)
        labels = torch.cat([self.transform(img[0]).view(-1).unsqueeze(0) for img in image_ls], dim=0)
        return images, labels
