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

from models.model_pretrain import ALBEF
from models.model_pretrain_simple import AlbefSimple
from models.vit import interpolate_pos_embed
from transformers import AutoTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, pretrain_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer


name_to_model = {
    'ALBEF': ALBEF,
    'AlbefSimple': AlbefSimple,
}


def train_epoch(args, model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    assert args.mode in ['train', 'both']
    # train
    model.train()

    metric_logger = utils.MetricLogger(f_path=os.path.join(config['output_path'], "log_metric.txt"), delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (images, input_ids, attn_masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_masks = attn_masks.to(device, non_blocking=True)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1.0, i / len(data_loader))

        loss_mlm, loss_ita, loss_itm = model(images, input_ids, attn_masks, 'train', alpha=alpha)

        loss = loss_mlm + loss_ita + loss_itm

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.log_global_avg()
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def train(args, config, model, train_loader, test_loader=None):
    device = torch.device(args.device)

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            if hasattr(model, 'visual_encoder_m'):
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                             model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        msg = model.load_state_dict(state_dict, strict=False)
        print('loading complete:', msg)
        print('load checkpoint from %s' % args.checkpoint)

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
                                  epoch, warmup_steps, device, lr_scheduler, config)
        # for validation
        test_stats = None
        if test_loader is not None:
            test_stats = test_epoch(args, model, test_loader, epoch, device, config)

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
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
def test_epoch(args, model, data_loader, epoch, device, config):
    assert args.mode in ['test', 'both']
    # test
    model.eval()

    metric_logger = utils.MetricLogger(
        f_path=os.path.join(config['output_path'], "log_test_metric.txt"), delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Test Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (images, input_ids, attn_masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_masks = attn_masks.to(device, non_blocking=True)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1.0, i / len(data_loader))

        loss_mlm, loss_ita, loss_itm = model(images, input_ids, attn_masks, 'test', alpha=alpha)

        loss = loss_mlm + loss_ita + loss_itm

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.log_global_avg()
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def test(args, config, model, data_loader):
    device = torch.device(args.device)

    # get all models
    if os.path.isdir(config['checkpoint_path']):
        model_list = sorted([m for m in os.listdir(config['checkpoint_path']) if m.endswith('.pth')])
    else:
        model_list = [config['checkpoint_path']]

    for cp_path in model_list:

        pos = cp_path.find('_')
        assert pos >= 0
        epoch = int(cp_path[(pos + 1):(pos + 3)])

        cp_path = os.path.join(config['checkpoint_path'], cp_path)
        # test every checkpoint
        checkpoint = torch.load(cp_path, map_location='cpu')
        state_dict = checkpoint['model']
        inner_model = model.module if hasattr(model, 'module') else model
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], inner_model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if hasattr(model, 'visual_encoder_m'):
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         inner_model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        msg = inner_model.load_state_dict(state_dict, strict=False)
        print('loading complete:', msg)
        print('load checkpoint from %s' % cp_path)

        start_time = time.time()

        test_stats = test_epoch(args, model, data_loader, epoch, device, config)
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
    datasets = [create_dataset('pretrain', mode, config, tokenizer)]

    if distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [mode == 'train'], num_tasks, global_rank)
    else:
        samplers = [None]

    def collate_fn(batch):
        return pretrain_collate_fn(batch, tokenizer)

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

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

    train_loader, test_loader = None, None
    if args.mode in ['train', 'both']:
        train_loader = init_dataset('train', config, args.distributed, tokenizer)
    if args.mode in ['test', 'both']:
        test_loader = init_dataset('test', config, args.distributed, tokenizer)

    # Model #
    print("Creating model")
    model = name_to_model[config['model']](config=config,
        text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True, distributed=args.distributed)

    model = model.to(torch.device(args.device))

    if args.mode != 'test':
        train(args, config, model, train_loader, test_loader)
    else:
        test(args, config, model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--text_encoder', default='../guwenbert-base')  # MODIFIED
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)  # MODIFIED
    parser.add_argument('--mode', choices=['train', 'test', 'both'], required=True)
    main_args = parser.parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main_config = yaml.load(open(main_args.config, 'r'), Loader=yaml.Loader)

    Path(main_config['output_path']).mkdir(parents=True, exist_ok=True)

    yaml.dump(main_config, open(os.path.join(main_config['output_path'], 'config.yaml'), 'w'))

    main(main_args, main_config)
