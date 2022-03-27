"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_single_mlm import SingleMlm
from transformers import AutoTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, mlm_single_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer


name_to_model = {
    'SingleMlm': SingleMlm,
}


def train_epoch(args, model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config, tokenizer=None):
    assert args.mode in ['train', 'both']
    # train
    model.train()

    save_cases, f_case = tokenizer is not None, None
    if save_cases:
        f_case = open(os.path.join(config['output_path'], 'logs_train',
                                   f'log_case_train_{epoch}.txt'), 'w', encoding='utf-8')

    metric_logger = utils.MetricLogger(f_path=os.path.join(config['output_path'], "log_metric.txt"), delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('correct_num', utils.SmoothedValue(
        window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
    metric_logger.add_meter('instance_num', utils.SmoothedValue(
        window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
    for k in config['topk']:
        metric_logger.add_meter(f'hit_correct_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
        metric_logger.add_meter(f'rank_correct_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:07}'))
        metric_logger.add_meter(f'rank_instance_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:07}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    data_idx = 0

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (images, input_ids, attn_masks, labels, pos_ids, type_ids, lengths, book_orders, mask_ids, mask_chs) \
            in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_masks = attn_masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pos_ids = pos_ids.to(device, non_blocking=True)
        type_ids = type_ids.to(device, non_blocking=True)
        mask_ids = mask_ids.to(device, non_blocking=True)

        loss_mlm, correct_num, instance_num, ori_inputs, correct_chars, wrong_chars, \
            rank_correct_num, rank_instance_num, hit_correct, topk_ids = \
            model(images, input_ids, attn_masks, labels, pos_ids, type_ids, lengths, mask_ids, mask_chs, 'train')

        loss_mlm.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(correct_num=int(correct_num))
        metric_logger.update(instance_num=int(instance_num))
        update_map = {}
        for k in config['topk']:
            update_map[f'hit_correct_{k}'] = hit_correct[k]
            update_map[f'rank_correct_{k}'] = rank_correct_num[k]
            update_map[f'rank_instance_{k}'] = rank_instance_num[k]
        metric_logger.update(**update_map)

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        if save_cases:
            for ch, idx in correct_chars:
                f_case.write(f'{tokenizer.convert_ids_to_tokens(ch)}\t{str(idx)}\n')
            f_case.write('\n')
            wrong_chars = [f'{tokenizer.convert_ids_to_tokens(ch)} {tokenizer.convert_ids_to_tokens(wch)} {str(idx)}'
                           for ch, wch, idx in wrong_chars]
            f_case.write('Wrong: ' + str(wrong_chars) + '\n\n')
            assert len(book_orders) == len(ori_inputs) == len(topk_ids)
            for sent, book_order, topk_id in zip(ori_inputs, book_orders, topk_ids):
                f_case.write(str(data_idx) + '\t' +
                             str(tokenizer.convert_ids_to_tokens(sent)) + '\t' + book_order + '\n')
                f_case.write('Topk: ' + str(tokenizer.convert_ids_to_tokens(topk_id)) + '\n')
                data_idx += 1
            f_case.write('------------------------------\n\n')
        f_case.flush()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.log_default_metric()
    meters = metric_logger.meters
    res = {k: meter.metric_fmt.format(meter.default_metric) for k, meter in meters.items()}
    res['global_accuracy'] = round(100 * meters['correct_num'].total / meters['instance_num'].total, 2)
    for k in config['topk']:
        res[f'global_hit_{k}'] = round(100 * meters[f'hit_correct_{k}'].total / meters['instance_num'].total, 2)
        res[f'global_rank_acc_{k}'] = round(
            100 * meters[f'rank_correct_{k}'].total / meters[f'rank_instance_{k}'].total, 2)

    if save_cases:
        f_case.close()

    return res


def train(args, config, model, train_loader, test_loader=None, tokenizer=None):
    device = torch.device(args.device)

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    cur_max_global_acc = 0.0

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

        msg = model.load_state_dict(state_dict, strict=False)
        cur_max_global_acc = checkpoint['global_accuracy']
        print('loading complete:', msg)
        print('load checkpoint from %s' % args.checkpoint)
        print('baseline accuracy ' + str(cur_max_global_acc))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train_epoch(args, model, train_loader, optimizer,
                                  epoch, warmup_steps, device, lr_scheduler, config, tokenizer=tokenizer)
        # for validation
        test_stats = None
        if test_loader is not None:
            test_stats = test_epoch(args, model, test_loader, epoch, device, config, tokenizer=tokenizer)
            if test_stats['global_accuracy'] > cur_max_global_acc:
                cur_max_global_acc = test_stats['global_accuracy']
                save_flag = True
                remove_flag = True
            else:
                save_flag = False
                remove_flag = False
        else:
            # train mode, save the newest
            save_flag = True
            remove_flag = False
        save_flag |= args.save_all
        remove_flag &= not args.save_all

        if utils.is_main_process():
            if remove_flag:
                os.system(f'rm -rf {config["output_path"]}/*.pth')
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            if save_flag:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'global_accuracy': cur_max_global_acc,
                }
                torch.save(save_obj, os.path.join(config['output_path'], 'checkpoint_%02d.pth' % epoch))

            print(log_stats)
            with open(os.path.join(config['output_path'], "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if test_stats is not None:
                log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}
                print(log_stats)
                with open(os.path.join(config['output_path'], "log_test.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def test_epoch(args, model, data_loader, epoch, device, config, tokenizer=None):
    assert args.mode in ['test', 'both']
    # test
    model.eval()

    save_cases, f_case = tokenizer is not None, None
    if save_cases:
        f_case = open(os.path.join(config['output_path'], 'logs_test',
                                   f'log_case_test_{epoch}.txt'), 'w', encoding='utf-8')

    metric_logger = utils.MetricLogger(
        f_path=os.path.join(config['output_path'], "log_test_metric.txt"), delimiter="  ")
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('correct_num', utils.SmoothedValue(
        window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
    metric_logger.add_meter('instance_num', utils.SmoothedValue(
        window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
    for k in config['topk']:
        metric_logger.add_meter(f'hit_correct_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:06}'))
        metric_logger.add_meter(f'rank_correct_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:07}'))
        metric_logger.add_meter(f'rank_instance_{k}', utils.SmoothedValue(
            window_size=50, fmt='{value:03}', metric='global_total', metric_fmt='{:07}'))

    header = 'Test Epoch: [{}]'.format(epoch)
    print_freq = 10
    data_idx = 0

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (images, input_ids, attn_masks, labels, pos_ids, type_ids, lengths, book_orders, mask_ids, mask_chs) \
            in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_masks = attn_masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pos_ids = pos_ids.to(device, non_blocking=True)
        type_ids = type_ids.to(device, non_blocking=True)
        mask_ids = mask_ids.to(device, non_blocking=True)

        loss_mlm, correct_num, instance_num, ori_inputs, correct_chars, wrong_chars, \
        rank_correct_num, rank_instance_num, hit_correct, topk_ids = \
            model(images, input_ids, attn_masks, labels, pos_ids, type_ids, lengths, mask_ids, mask_chs, 'test')

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(correct_num=int(correct_num))
        metric_logger.update(instance_num=int(instance_num))
        update_map = {}
        for k in config['topk']:
            update_map[f'hit_correct_{k}'] = hit_correct[k]
            update_map[f'rank_correct_{k}'] = rank_correct_num[k]
            update_map[f'rank_instance_{k}'] = rank_instance_num[k]
        metric_logger.update(**update_map)

        if save_cases:
            for ch, idx in correct_chars:
                f_case.write(f'{tokenizer.convert_ids_to_tokens(ch)}\t{str(idx)}\n')
            f_case.write('\n')
            wrong_chars = [f'{tokenizer.convert_ids_to_tokens(ch)} {tokenizer.convert_ids_to_tokens(wch)} {str(idx)}'
                           for ch, wch, idx in wrong_chars]
            f_case.write('Wrong: ' + str(wrong_chars) + '\n\n')
            assert len(book_orders) == len(ori_inputs) == len(topk_ids)
            for sent, book_order, topk_id in zip(ori_inputs, book_orders, topk_ids):
                f_case.write(str(data_idx) + '\t' +
                             str(tokenizer.convert_ids_to_tokens(sent)) + '\t' + book_order + '\n')
                f_case.write('Topk: ' + str(tokenizer.convert_ids_to_tokens(topk_id)) + '\n')
                data_idx += 1
            f_case.write('------------------------------\n\n')
        f_case.flush()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.log_default_metric()
    meters = metric_logger.meters
    res = {k: meter.metric_fmt.format(meter.default_metric) for k, meter in meters.items()}
    res['global_accuracy'] = round(100 * meters['correct_num'].total / meters['instance_num'].total, 2)
    for k in config['topk']:
        res[f'global_hit_{k}'] = round(100 * meters[f'hit_correct_{k}'].total / meters['instance_num'].total, 2)
        res[f'global_rank_acc_{k}'] = round(
            100 * meters[f'rank_correct_{k}'].total / meters[f'rank_instance_{k}'].total, 2)

    if save_cases:
        f_case.close()

    return res


def test(args, config, model, data_loader, tokenizer=None):
    device = torch.device(args.device)

    # get all models
    if os.path.isdir(args.checkpoint):
        model_list = sorted([m for m in os.listdir(args.checkpoint) if m.endswith('.pth')])
    else:
        model_list = [args.checkpoint]

    for cp_path in model_list:

        pos = cp_path.find('_')
        assert pos >= 0
        epoch = int(cp_path[(pos + 1):(pos + 3)])

        cp_path = os.path.join(args.checkpoint, cp_path)
        # test every checkpoint
        checkpoint = torch.load(cp_path, map_location='cpu')
        state_dict = checkpoint['model']
        inner_model = model.module if hasattr(model, 'module') else model
        msg = inner_model.load_state_dict(state_dict, strict=False)
        print('loading complete:', msg)
        print('load checkpoint from %s' % cp_path)

        start_time = time.time()

        test_stats = test_epoch(args, model, data_loader, epoch, device, config, tokenizer=tokenizer)
        if utils.is_main_process():
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}

            print(log_stats)
            with open(os.path.join(config['output_path'], "log_test.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Testing time {} for epoch {}'.format(total_time_str, epoch))


def init_dataset(mode, config, distributed, tokenizer):
    # Dataset #
    print("Creating dataset")
    datasets = [create_dataset('finetune_single_mlm', mode, config, tokenizer)]

    if distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [mode == 'train'], num_tasks, global_rank)
    else:
        samplers = [None]
    modality = config['modality'] if 'modality' in config else 'cross'
    assert modality in ['cross', 'text', 'image']

    def collate_fn(batch):
        return mlm_single_collate_fn(batch, tokenizer, modality)

    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']],
                                num_workers=[4], is_trains=[mode == 'train'], collate_fns=[collate_fn])[0]
    return data_loader


def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)

    train_loader, test_loader = None, None
    if args.mode in ['train', 'both']:
        train_loader = init_dataset('train', config, args.distributed, tokenizer)
    if args.mode in ['test', 'both']:
        test_loader = init_dataset('test', config, args.distributed, tokenizer)

    # Model #
    print("Creating model")
    model = name_to_model[config['model']](config=config,
        text_encoder=args.text_encoder, tokenizer=tokenizer, distributed=args.distributed)

    model = model.to(torch.device(args.device))

    if args.mode != 'test':
        train(args, config, model, train_loader, test_loader, tokenizer)
    else:
        test(args, config, model, test_loader, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Finetune_single_mlm.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--text_encoder', default='')  # MODIFIED
    parser.add_argument('--text_tokenizer', default='../guwenbert-base')  # MODIFIED
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)  # MODIFIED
    parser.add_argument('--mode', choices=['train', 'test', 'both'], required=True)
    parser.add_argument('--save_all', default=True, type=bool)
    main_args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main_config = yaml.load(open(main_args.config, 'r'), Loader=yaml.Loader)

    Path(main_config['output_path']).mkdir(parents=True, exist_ok=True)
    if main_args.mode in ['train', 'both']:
        os.makedirs(os.path.join(main_config['output_path'], 'logs_train'), exist_ok=True)
    if main_args.mode in ['test', 'both']:
        os.makedirs(os.path.join(main_config['output_path'], 'logs_test'), exist_ok=True)

    yaml.dump(main_config, open(os.path.join(main_config['output_path'], 'config.yaml'), 'w'))

    main(main_args, main_config)
