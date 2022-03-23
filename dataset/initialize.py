import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from dataset.randaugment import RandomAugment
from dataset.caption_dataset import re_train_dataset, re_eval_dataset
from dataset.complete_dataset import OracleCompleteDataset, OracleCompleteSingleDataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset


def create_dataset(dataset, mode, config, tokenizer):
    # Image.open: height * width * channel
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain':
        # MODIFIED
        dataset = OracleCompleteDataset(config, mode, tokenizer, add_mask=False)
        # dataset = pretrain_dataset(config, mode, pretrain_transform)
        return dataset

    elif dataset == 'finetune_mlm':
        dataset = OracleCompleteDataset(config, mode, tokenizer, add_mask=True)
        return dataset

    elif dataset == 'finetune_single_mlm':
        dataset = OracleCompleteSingleDataset(config, mode, tokenizer, add_mask=True)
        return dataset

    elif dataset == 're':
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'vqa':
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'],
                                    split='train')
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'])
        return train_dataset, vqa_test_dataset

    elif dataset == 'nlvr':
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 've':
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'grounding':
        train_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')
        return train_dataset, test_dataset
    else:
        raise ValueError("create_dataset name error")


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


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
    images, sentences, attn_masks, labels, pos_ids, type_ids, lengths = [], [], [], [], [], [], []
    # longest value n1
    max_len = max(len(img) for img, _, _ in batch)
    for img, input_ids, label in batch:
        assert len(img) + 2 == len(input_ids) == len(label)
        n1, n2 = len(img), max_len - len(img)
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

    return torch.cat(images, dim=0), input_ids, torch.FloatTensor(attn_masks), \
        torch.cat(labels, dim=0), torch.LongTensor(pos_ids), torch.LongTensor(type_ids), lengths


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
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
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
