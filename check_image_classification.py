import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', help='model name', type=str)
    parser.add_argument('--epoch', '-e', help='epoch number', type=int)
    parser.add_argument('--head', help='print number', type=int, default=20)
    args = parser.parse_args()

    fin = open(f'output/{args.task}/logs_test/log_case_test_{args.epoch:02}.txt', encoding='utf-8')
    lines = [line.strip().split('\t') for line in fin.readlines()]
    lines = [(int(lab), int(res), image_p) for lab, res, image_p in lines]

    counter = {}
    for lab, res, image_p in lines:
        assert res in [0, 1]
        counter.setdefault(lab, [0, 0])
        counter[lab][0] += 1
        counter[lab][1] += 1 - res
    counter = sorted(list(counter.items()), key=lambda x: x[1][1], reverse=True)
    print(counter[:args.head])

    print('------------')
    label_to_char = json.load(open('../simclr/orcal/label_to_char.json', encoding='utf-8'))
    label_to_char = {int(key): val for key, val in label_to_char.items()}
    print([label_to_char[lab] for lab, cnt in counter[:args.head]])


if __name__ == '__main__':
    main()
