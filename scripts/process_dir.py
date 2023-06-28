import os
import argparse
from utils import load_json, save_json


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


def main2():
    input_dir = 'handa/H32384_detection_with_label'
    images = [img for img in os.listdir(input_dir) if img.endswith('.jpg')]
    images.sort(key=lambda x: int(x.replace('-', '_').split('_')[2]))

    with open(f'{input_dir}/labels.txt', encoding='utf-8') as fin:
        lines = fin.readlines()
    assert len(lines) == len(images)
    ret = []
    all_chars, characters = [], []
    chat_to_label = load_json('handa/oracle_classification_chant_char_to_label.json')
    for idx, (img, line) in enumerate(zip(images, lines)):
        line = line.strip()
        to_complete = '——' in line
        ch = line.split('——')[0]
        img = os.path.join(input_dir, img)
        all_chars.append((img, ch, to_complete, idx))
        characters.append((ch, img))
        if to_complete:
            ret.append({'img': img, 'src': 'other', 'lab': chat_to_label[ch]})
    save_json([ret], 'handa/H32384_classification_with_label.json')
    ret = []
    for img, ch, to_complete, idx in all_chars:
        if to_complete:
            ret.append(({
                'book_name': 'H32384', 'row_order': 0, 'characters': characters,
            }, idx))
    save_json(ret, 'handa/H32384_complete_with_label.json')


if __name__ == '__main__':
    # main()
    main2()
