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

from models.unet import UNet

import utils
from dataset import create_dataset, create_sampler, create_loader, sharpen_unet_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer
from tqdm import tqdm
from PIL import Image


name_to_model = {
    'unet': UNet,
}


def train_epoch(args, model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    assert args.mode in ['train', 'train_valid']
    # train
    model.train()

    metric_logger = utils.MetricLogger(f_path=os.path.join(config['output_path'], "log_metric.txt"), delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()

        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device, non_blocking=True)

        loss = model(data_dict, 'train')

        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if config['scale']:
            metric_logger.update(n=config['batch_size'], loss=loss.item() * 1e4)
        else:
            metric_logger.update(n=config['batch_size'], loss=loss.item())

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.log_default_metric()
    meters = metric_logger.meters
    res = {k: meter.metric_fmt.format(meter.default_metric) for k, meter in meters.items()}

    return res


def train(args, config, model, train_loader, valid_loader=None):
    device = torch.device(args.device)

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    cur_max_global_loss = float('inf')

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

        msg = model.load_state_dict(state_dict, strict=False)
        cur_max_global_loss = checkpoint['global_loss']
        print('loading complete:', msg)
        print('load checkpoint from %s' % args.checkpoint)
        print('baseline loss ' + str(cur_max_global_loss))

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
        valid_stats = None
        if valid_loader is not None:
            valid_stats = valid_epoch(args, model, valid_loader, epoch, device, config)
            if valid_stats['global_loss'] < cur_max_global_loss:
                cur_max_global_loss = valid_stats['global_loss']
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
                    'global_loss': cur_max_global_loss,
                }
                torch.save(save_obj, os.path.join(config['output_path'], 'checkpoint_%02d.pth' % epoch))

            print(log_stats)
            with open(os.path.join(config['output_path'], "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if valid_stats is not None:
                log_stats = {**{f'test_{k}': v for k, v in valid_stats.items()}, 'epoch': epoch}
                print(log_stats)
                with open(os.path.join(config['output_path'], "log_valid.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def valid_epoch(args, model, data_loader, epoch, device, config):
    assert args.mode == 'train_valid'
    # test
    model.eval()

    metric_logger = utils.MetricLogger(
        f_path=os.path.join(config['output_path'], "log_valid_metric.txt"), delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Test Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device, non_blocking=True)

        loss = model(data_dict, 'valid')

        if config['scale']:
            metric_logger.update(n=config['batch_size'], loss=loss.item() * 1e4)
        else:
            metric_logger.update(n=config['batch_size'], loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.log_default_metric()
    meters = metric_logger.meters
    res = {k: meter.metric_fmt.format(meter.default_metric) for k, meter in meters.items()}

    return res


@torch.no_grad()
def test(args, config, model, data_loader):
    device = torch.device(args.device)

    assert args.checkpoint != '' and not os.path.isdir(args.checkpoint)

    cp_path = os.path.basename(args.checkpoint)
    pos = cp_path.find('_')
    assert pos >= 0
    epoch = int(cp_path[(pos + 1):(pos + 3)])

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    inner_model = model.module if hasattr(model, 'module') else model
    msg = inner_model.load_state_dict(state_dict, strict=False)
    print('loading complete:', msg)
    print('load checkpoint from %s' % cp_path)

    start_time = time.time()

    model.eval()

    pbar = tqdm(range(len(data_loader)))

    for step, data_dict in enumerate(data_loader):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device, non_blocking=True)

        result = model(data_dict, 'test')
        output_img, output_path = result['output_img'], result['output_path']
        assert output_img.shape[0] == len(output_path)
        for iid in range(len(output_path)):
            # save images
            img, img_path = output_img[iid].cpu().numpy(), output_path[iid]
            img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 1:
                img = np.squeeze(img, axis=2)
            img = Image.fromarray(img.astype(np.uint8), config['img_mode'])
            img.save(img_path)
        pbar.update()

    pbar.close()

    if args.distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {} for epoch {}'.format(total_time_str, epoch))


global_loader_generator = torch.Generator()


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_dataset(mode, config, distributed):
    # Dataset #
    global global_loader_generator
    print("Creating dataset")
    datasets = [create_dataset('sharpen_unet', mode, config)]

    if distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [mode == 'train'], num_tasks, global_rank)
    else:
        samplers = [None]

    def collate_fn(batch):
        return sharpen_unet_collate_fn(batch, config, mode)

    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']],
                                num_workers=[4], is_trains=[mode == 'train'], collate_fns=[collate_fn],
                                worker_init_fn=seed_worker, generator=global_loader_generator)[0]
    return data_loader


def init_test_config(args, config):
    # set input_path and output_path
    if args.mode != 'test':
        return
    if isinstance(config['data_path']['test'], list):
        assert len(config['data_path']['test']) == 2
        test_input, test_output = config['data_path']['test']
        if args.input_path is not None:
            test_input = args.input_path
        if args.output_path is not None:
            test_output = args.output_path
        config['data_path']['test'] = [test_input, test_output]
        assert test_input is not None and test_output is not None
        os.makedirs(test_output, exist_ok=True)
    else:
        test_input = config['data_path']['test']
        if args.input_path is not None:
            test_input = args.input_path
        assert test_input is not None
        config['data_path']['test'] = test_input


def main(args, config):
    global global_loader_generator
    if args.distributed:
        utils.init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    global_loader_generator.manual_seed(seed)
    cudnn.benchmark = True

    init_test_config(args, config)

    train_loader, test_loader = None, None
    if args.mode in ['train', 'train_valid']:
        train_loader = init_dataset('train', config, args.distributed)
        if args.mode == 'train_valid':
            test_loader = init_dataset('valid', config, args.distributed)
    else:
        test_loader = init_dataset('test', config, args.distributed)

    # Model #
    print("Creating model")
    model = name_to_model[config['model']](config=config)

    model = model.to(torch.device(args.device))

    if args.mode != 'test':
        train(args, config, model, train_loader, test_loader)
    else:
        test(args, config, model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Sharpen_unet.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--input_path', help='path of the test input folder', default=None)
    parser.add_argument('--output_path', help='path of the test output folder', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)  # MODIFIED
    parser.add_argument('--mode', choices=['train', 'train_valid', 'test'], required=True)
    parser.add_argument('--save_all', default=True, type=bool)
    main_args = parser.parse_args()

    main_config = yaml.load(open(main_args.config, 'r'), Loader=yaml.Loader)
    Path(main_config['output_path']).mkdir(parents=True, exist_ok=True)
    yaml.dump(main_config, open(os.path.join(main_config['output_path'], 'config.yaml'), 'w'))

    main(main_args, main_config)
