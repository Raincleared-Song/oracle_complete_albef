import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', help='model name', type=str)
    parser.add_argument('--metric', '-m', help='metric', default='valid_global_accuracy', type=str)
    parser.add_argument('--reverse', '-r', help='if set, choose the lowest metric', action='store_true')
    parser.add_argument('--remove', '-rm', help='if set, remove redundant checkpoints', action='store_true')
    args = parser.parse_args()

    checkpoints = os.listdir(f'output/{args.task}')
    metric_file = 'log_test.txt' if 'log_test.txt' in checkpoints else 'log_valid.txt'
    checkpoints = sorted([cp for cp in checkpoints if cp.startswith('checkpoint_')], key=lambda x: int(x[11:-4]))
    print('got checkpoints count:', len(checkpoints))

    with open(f'output/{args.task}/{metric_file}') as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    chosen_metric, chosen_line = float('inf') if args.reverse else -1., {}
    for line in lines:
        line = json.loads(line)
        assert args.metric in line and 'epoch' in line
        metric, epoch = line[args.metric], line['epoch']
        if isinstance(metric, str):
            metric = float(metric)
        if args.reverse and metric < chosen_metric or not args.reverse and metric > chosen_metric:
            chosen_metric = metric
            chosen_line = line

    assert chosen_line != {}
    print(chosen_line)
    print('chosen metric:', chosen_line[args.metric], 'chosen_epoch:', chosen_line['epoch'],
          'last epoch:', int(checkpoints[-1][11:-4]))

    if args.remove:
        best_epoch, last_epoch = chosen_line['epoch'], int(checkpoints[-1][-6:-4])
        os.system(f'mv output/{args.task}/checkpoint_{best_epoch:02}.pth output/')
        if last_epoch != best_epoch:
            os.system(f'mv output/{args.task}/checkpoint_{last_epoch:02}.pth output/')
        os.system(f'rm -rf output/{args.task}/checkpoint_*.pth')
        os.system(f'mv output/checkpoint_*.pth output/{args.task}/')


if __name__ == '__main__':
    main()
