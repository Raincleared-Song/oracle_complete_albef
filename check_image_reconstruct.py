import torch
import argparse
from dataset.data_utils import tensor_to_img


def parse_indexes(indexes: str):
    indexes = indexes.split(',')
    res = set()
    for index in indexes:
        if '-' in index:
            tokens = index.split('-')
            assert len(tokens) == 2
            begin, end = int(tokens[0]), int(tokens[1])
            assert begin <= end
            res |= set(range(begin, end + 1))
        else:
            res.add(int(index))
    return sorted(list(res))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', help='model name', type=str)
    parser.add_argument('--index', '-i', help='pth indexes', default='0', type=str)
    parser.add_argument('--sub_index', '-s', help='sub image indexes', default='0', type=str)
    parser.add_argument('--save_all', '-sa', help='if set, save all images', action='store_true')
    args = parser.parse_args()

    mean, std = [0.5601, 0.5598, 0.5596], [0.4064, 0.4065, 0.4066]
    trans, total = 0, 0

    for index in parse_indexes(args.index):
        embeds, targets, images = torch.load(f'output/{args.task}/{index}.pth')
        for sub_index in parse_indexes(args.sub_index):
            save_flag = torch.sum(targets[sub_index]) != torch.sum(images[sub_index])
            total += 1
            trans += int(save_flag)
            if save_flag or args.save_all:
                img = tensor_to_img(embeds[sub_index], (3, 128, 128), mean, std)
                img.save(f'output/{args.task}/test_{index}{sub_index}_embed.png')
                img = tensor_to_img(targets[sub_index], (3, 128, 128), mean, std)
                img.save(f'output/{args.task}/test_{index}{sub_index}_target.png')
                img = tensor_to_img(images[sub_index], (3, 128, 128), mean, std)
                img.save(f'output/{args.task}/test_{index}{sub_index}_input.png')
    print('transformed:', trans, 'total:', total)


if __name__ == '__main__':
    main()
