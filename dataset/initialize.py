import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataset.data_utils import resize_pad_image
from dataset.complete_dataset import OracleCompleteDataset, OracleCompleteSingleDataset
from dataset.sharpen_dataset import SharpenDataset
from dataset.reconstruct_dataset import ImageReconstructDataset
from dataset.classification_dataset import ImageClassificationDataset


def create_dataset(dataset, mode, config, tokenizer=None):
    if dataset == 'pretrain':
        dataset = OracleCompleteDataset(config, mode, tokenizer, add_mask=False)
        return dataset

    elif dataset == 'finetune_mlm':
        dataset = OracleCompleteDataset(config, mode, tokenizer, add_mask=True)
        return dataset

    elif dataset == 'finetune_single_mlm':
        dataset = OracleCompleteSingleDataset(config, mode, tokenizer, add_mask=True)
        return dataset

    elif dataset == 'sharpen_unet':
        dataset = SharpenDataset(config, mode)
        return dataset

    elif dataset == 'image_reconstruct':
        dataset = ImageReconstructDataset(config, mode, tokenizer)
        return dataset

    elif dataset == 'image_classification':
        dataset = ImageClassificationDataset(config, mode)
        return dataset

    else:
        raise ValueError("create_dataset name error")


def pretrain_collate_fn(batch, tokenizer):
    images, sentences, attn_masks = [], [], []
    # padding to longest
    max_len = max(len(input_ids) for img, input_ids in batch)
    for img, input_ids in batch:
        images.append(img.unsqueeze(0))
        ori_len, pad_len = len(input_ids), max_len - len(input_ids)
        input_ids = torch.cat((input_ids, torch.LongTensor([tokenizer.pad_token_id] * pad_len)), dim=0)
        attn_mask = [1] * ori_len + [0] * pad_len
        sentences.append(input_ids.unsqueeze(0))
        attn_masks.append(attn_mask)
    return torch.cat(images, dim=0), torch.cat(sentences, dim=0), torch.FloatTensor(attn_masks)


def mlm_collate_fn(batch, tokenizer):
    images, sentences, attn_masks, labels = [], [], [], []
    # padding to longest
    max_len = max(len(input_ids) for img, input_ids, label in batch)
    for img, input_ids, label in batch:
        images.append(img.unsqueeze(0))
        ori_len, pad_len = len(input_ids), max_len - len(input_ids)
        assert len(label) == len(input_ids)
        input_ids = torch.cat((input_ids, torch.LongTensor([tokenizer.pad_token_id] * pad_len)), dim=0)
        label = torch.cat((label, torch.LongTensor([-100] * pad_len)), dim=0)
        attn_mask = [1] * ori_len + [0] * pad_len
        sentences.append(input_ids.unsqueeze(0))
        labels.append(label.unsqueeze(0))
        attn_masks.append(attn_mask)
    return torch.cat(images, dim=0), \
        torch.cat(sentences, dim=0), torch.FloatTensor(attn_masks), torch.cat(labels, dim=0)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def mlm_single_collate_fn(batch, tokenizer, modality, img_pad_color=1.0):
    """
    for finetune_mlm_single
    [CLS], text tokens * n1, [SEP], [PAD] * n2, image embeds * n1, [IMG_PAD] * n2
    return: image_matrix with padding, input_ids with cls+seq+padding, attn_mask (both), labels (both)
    """
    images, sentences, attn_masks, labels, pos_ids, type_ids, lengths, book_orders = [], [], [], [], [], [], [], []
    mask_ids, mask_chs, mask_img_ids, mask_ori_images = [], [], [], []
    # longest value n1
    max_len = max(len(img) for img, _, _, _, _, _, _ in batch)
    for img, input_ids, label, book_order, mask_id, mask_ch, mask_ori_img in batch:
        assert len(img) + 2 == len(input_ids) == len(label)
        book_orders.append(book_order)
        mask_chs.append(mask_ch)
        n1, n2 = len(img), max_len - len(img)
        if modality == 'image':
            mask_ids.append(mask_id - 1)
            mask_img_ids.append(mask_id - 1)
        elif modality == 'text':
            mask_ids.append(mask_id)
            mask_img_ids.append(mask_id)
        else:
            mask_ids.append(mask_id)
            mask_img_ids.append(1 + n1 + n2 + mask_id)
        # [batch_size, n1 + n2, res*res*chan]
        img = torch.cat((img, torch.full((n2, img.shape[1]), img_pad_color)), dim=0)
        images.append(img.unsqueeze(0))
        # [batch_size, cls + n1 + sep + n2]
        input_ids = torch.cat((input_ids, torch.LongTensor([tokenizer.pad_token_id] * n2)), dim=0)
        sentences.append(input_ids.unsqueeze(0))
        if modality == 'cross':
            # [batch_size, cls + n1 + sep + n2 + n1 + n2]
            label = torch.cat((label, torch.LongTensor([-100] * (max_len + n2))), dim=0)
            attn_mask = [1] * (n1 + 2) + [0] * n2 + [1] * n1 + [0] * n2
        elif modality == 'text':
            label = torch.cat((label, torch.LongTensor([-100] * n2)), dim=0)
            attn_mask = [1] * (n1 + 2) + [0] * n2
        else:
            label = torch.cat((label[1:-1], torch.LongTensor([-100] * n2)), dim=0)
            attn_mask = [1] * n1 + [0] * n2
        labels.append(label.unsqueeze(0))
        attn_masks.append(attn_mask)

        lengths.append((n1, n2))
        # assert len(label) == len(attn_mask) == len(img) + len(input_ids)
        mask_ori_images.append(mask_ori_img)

    input_ids = torch.cat(sentences, dim=0)
    temp_pos = create_position_ids_from_input_ids(input_ids, tokenizer.pad_token_id)
    for bid, (n1, n2) in enumerate(lengths):
        # cls + n1 + sep + n2
        cur_pos = temp_pos[bid, :]
        assert cur_pos.ndim == 1 and len(cur_pos) == n1 + n2 + 2
        cur_pos = cur_pos.tolist()
        cls_pos, word_pos, pad_pos = cur_pos[:1], cur_pos[1:(n1+2)], cur_pos[(n1+2):]
        if modality == 'cross':
            # cls + n1 + sep + n2 + n1 + n2
            pos_ids.append(cls_pos + word_pos + pad_pos + word_pos[:-1] + pad_pos)
            type_ids.append([0] * (n1 + n2 + 2) + [1] * (n1 + n2))
        elif modality == 'text':
            pos_ids.append(cls_pos + word_pos + pad_pos)
            type_ids.append([0] * (n1 + n2 + 2))
        else:
            pos_ids.append(word_pos[:-1] + pad_pos)
            type_ids.append([1] * (n1 + n2))

    assert len(mask_ids) == len(mask_chs) == len(mask_img_ids)
    for mask_id, mask_ch, mask_img_id in zip(mask_ids, mask_chs, mask_img_ids):
        assert 0 <= mask_id <= mask_img_id and mask_ch >= 0

    return torch.cat(images, dim=0), torch.cat(mask_ori_images, dim=0), input_ids, torch.FloatTensor(attn_masks), \
        torch.cat(labels, dim=0), torch.LongTensor(pos_ids), torch.LongTensor(type_ids), \
        lengths, book_orders, torch.LongTensor(mask_ids), torch.LongTensor(mask_img_ids), mask_chs


def image_reconstruct_collate_fn(batch, img_pad_color=1.0):
    images, labels, texts = [], [], []
    max_len = max(len(img) for img, _, _ in batch)
    for image, label, text in batch:
        assert len(image) == len(label) == len(text)
        pad_len = max_len - len(image)
        image = torch.cat((image, torch.full((pad_len, image.shape[1]), img_pad_color)), dim=0)
        images.append(image.unsqueeze(0))
        label = torch.cat((label, torch.full((pad_len, label.shape[1]), img_pad_color)), dim=0)
        labels.append(label.unsqueeze(0))
        texts.append(text + [-100] * pad_len)
    return torch.cat(images, dim=0), torch.cat(labels, dim=0), torch.LongTensor(texts)


def image_classification_collate_fn(batch):
    images, labels, image_paths = [], [], []
    for image, label, image_p in batch:
        images.append(image)
        labels.append(label)
        image_paths.append(image_p)
    return torch.cat(images, dim=0), torch.LongTensor(labels), image_paths


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns,
                  worker_init_fn=None, generator=None):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            # pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        loaders.append(loader)
    return loaders


def process_sharpen_single(config, img_path):
    """img_path -> (channel_num, height, width)"""
    img = Image.open(img_path).convert(config['img_mode'])
    pad_shape = config['image_res'], config['image_res']
    img = resize_pad_image(img, pad_shape, do_trans=config['img_random_transform'],
                           pad_color=config['pad_color'], mask_ratio=config['img_mask_ratio'],
                           noise_ratio=config['img_noise_ratio'], do_rotate=config['img_do_rotate'])
    assert img.ndim in (2, 3)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)
        assert img.shape == (1,) + pad_shape, str(img.shape)
    else:
        assert img.shape[2] <= 4
        img = np.transpose(img, (2, 0, 1))
        assert img.shape[1:] == pad_shape, str(img.shape)
    if config['scale']:
        img = img.astype(np.float) / 256.0
    return img


def sharpen_unet_collate_fn(batch, config, mode):
    noise_batch, label_batch = [], []
    pad_shape = len(config['img_mode']), config['image_res'], config['image_res']
    for noise_path, label_path in batch:
        noise_img = process_sharpen_single(config, noise_path)
        if mode != 'test':
            label_img = process_sharpen_single(config, label_path)
            assert noise_img.shape == label_img.shape == pad_shape, str(noise_img.shape) + ' ' + str(label_img.shape)
        else:
            label_img = label_path
            assert noise_img.shape == pad_shape
        noise_batch.append(noise_img)
        label_batch.append(label_img)
    return {
        'inputs': torch.FloatTensor(noise_batch),
        'labels': torch.FloatTensor(label_batch) if mode != 'test' else label_batch
    }
