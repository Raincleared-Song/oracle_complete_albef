import os
import argparse
from utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='handa/H32384_detection_result')
    args = parser.parse_args()

    images = [img for img in os.listdir(args.input_dir) if img.endswith('.jpg')]
    images.sort(key=lambda x: int(x.replace('-', '_').split('_')[2]))

    ret, all_images = [], []
    for img in images:
        ret.append({'img': os.path.join(args.input_dir, img), 'src': 'other', 'lab': 0})
        all_images.append(os.path.join(args.input_dir, img))
    save_json([ret], 'handa/H32384_classification.json')

    with open('handa/H32384_detection_result/log_case_test_2.txt', encoding='utf-8') as fin:
        chars = [line.split('\t')[2] for line in fin.readlines()]
        print(chars)
    assert len(chars) == len(images)
    all_chars = [(ch, img) for ch, img in zip(chars, all_images)]
    com_ret = []
    for idx in range(len(all_chars)):
        com_ret.append(({
            'book_name': 'H32384', 'row_order': 0, 'characters': all_chars,
        }, idx))
    save_json(com_ret, 'handa/H32384_complete.json')


if __name__ == '__main__':
    main()
