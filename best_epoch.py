import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', help='model name', type=str)
    parser.add_argument('--metric', '-m', help='metric', default='valid_global_accuracy', type=str)
    parser.add_argument('--reverse', '-r', help='if set, choose the lowest metric', action='store_true')
    parser.add_argument('--remove', '-rm', help='if set, remove redundant checkpoints', action='store_true')
    parser.add_argument('--device', '-d', help='the device for test', type=str, default='cuda:0')
    parser.add_argument('--use_test', action='store_true', help='if set, use log_test.txt')
    parser.add_argument('--double', action='store_true', help='if set, test both do_trans')
    parser.add_argument('--validation', action='store_true', help='if set, test on validation set')
    parser.add_argument('--check_only', action='store_true', help='if set, do not conduct any test')
    parser.add_argument('--test_only', action='store_true', help='if set, only test on combined test set')
    args = parser.parse_args()

    checkpoints = os.listdir(f'output/{args.task}')
    metric_file = 'log_test.txt' if args.use_test else 'log_valid.txt'
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

    if args.metric == 'valid_global_accuracy' and not args.task.startswith('image_class'):
        hits = []
        for k in [1, 5, 10, 20]:
            hit = round(chosen_line[f'valid_global_hit_{k}'], 2)
            hits.append(f'{hit:.2f}')
        print(' / '.join(hits))

    best_epoch, last_epoch = chosen_line['epoch'], int(checkpoints[-1][11:-4])
    if args.remove:
        os.system(f'mv output/{args.task}/checkpoint_{best_epoch:02}.pth output/')
        if last_epoch != best_epoch:
            os.system(f'mv output/{args.task}/checkpoint_{last_epoch:02}.pth output/')
        os.system(f'rm -rf output/{args.task}/checkpoint_*.pth')
        os.system(f'mv output/checkpoint_*.pth output/{args.task}/')

    if args.check_only:
        return

    # 合并后的完整测试集，1648
    print('------', 'testing 1648 files do_trans false', '------')
    assert os.system(f'python Finetune_single_mlm.py --config output/{args.task}/config.yaml --device {args.device}'
                     f' --checkpoint output/{args.task}/checkpoint_{best_epoch:02}.pth --mode test'
                     f' --test_files handa/cases_com_tra_mid_combine.json --do_trans false') == 0
    assert os.system(f'mv output/{args.task}/logs_test/log_case_test_{best_epoch}.txt'
                     f' output/{args.task}/logs_test/log_case_test_{best_epoch}_1648.txt') == 0

    if args.test_only:
        return

    # 验证集
    if args.validation:
        print('------', 'testing log_test_52 do_trans false', '------')
        assert os.system(f'python Finetune_single_mlm.py --config output/{args.task}/config.yaml --device {args.device}'
                         f' --checkpoint output/{args.task}/checkpoint_{best_epoch:02}.pth --mode test'
                         f' --test_files handa/log_case_test_52_data.json --do_trans false') == 0

    # 训练集残字，1067
    if args.double:
        print('------', 'testing 1067 files do_trans true', '------')
        assert os.system(f'python Finetune_single_mlm.py --config output/{args.task}/config.yaml --device {args.device}'
                         f' --checkpoint output/{args.task}/checkpoint_{best_epoch:02}.pth --mode test'
                         f' --test_files handa/cases_com_tra_mid_new.json --do_trans true') == 0

    print('------', 'testing 1067 files do_trans false', '------')
    assert os.system(f'python Finetune_single_mlm.py --config output/{args.task}/config.yaml --device {args.device}'
                     f' --checkpoint output/{args.task}/checkpoint_{best_epoch:02}.pth --mode test'
                     f' --test_files handa/cases_com_tra_mid_new.json --do_trans false') == 0
    assert os.system(f'mv output/{args.task}/logs_test/log_case_test_{best_epoch}.txt'
                     f' output/{args.task}/logs_test/log_case_test_{best_epoch}_1067.txt') == 0

    # 上下文推断，残字部分（小），156
    if args.double:
        print('------', 'testing 156 files do_trans true', '------')
        assert os.system(f'python Finetune_single_mlm.py --config output/{args.task}/config.yaml --device {args.device}'
                         f' --checkpoint output/{args.task}/checkpoint_{best_epoch:02}.pth --mode test'
                         f' --test_files handa/cases_com_tra_mid_new_not_perfect.json --do_trans true') == 0

    print('------', 'testing 156 files do_trans false', '------')
    assert os.system(f'python Finetune_single_mlm.py --config output/{args.task}/config.yaml --device {args.device}'
                     f' --checkpoint output/{args.task}/checkpoint_{best_epoch:02}.pth --mode test'
                     f' --test_files handa/cases_com_tra_mid_new_not_perfect.json --do_trans false') == 0
    assert os.system(f'mv output/{args.task}/logs_test/log_case_test_{best_epoch}.txt'
                     f' output/{args.task}/logs_test/log_case_test_{best_epoch}_156.txt') == 0

    # 汉达方框字，525
    print('------', 'testing 525 files do_trans false', '------')
    assert os.system(f'python Finetune_single_mlm.py --config output/{args.task}/config.yaml --device {args.device}'
                     f' --checkpoint output/{args.task}/checkpoint_{best_epoch:02}.pth --mode test'
                     f' --test_files handa/data_filter_tra_all_com_mid.json --do_trans false') == 0
    assert os.system(f'mv output/{args.task}/logs_test/log_case_test_{best_epoch}.txt'
                     f' output/{args.task}/logs_test/log_case_test_{best_epoch}_525.txt') == 0

    # 上下文推断，完整版，586
    print('------', 'testing 586 files do_trans false', '------')
    assert os.system(f'python Finetune_single_mlm.py --config output/{args.task}/config.yaml --device {args.device}'
                     f' --checkpoint output/{args.task}/checkpoint_{best_epoch:02}.pth --mode test'
                     f' --test_files handa/cases_com_tra_mid.json --do_trans false') == 0
    assert os.system(f'mv output/{args.task}/logs_test/log_case_test_{best_epoch}.txt'
                     f' output/{args.task}/logs_test/log_case_test_{best_epoch}_586.txt') == 0


if __name__ == '__main__':
    main()
