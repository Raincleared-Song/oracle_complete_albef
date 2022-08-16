import os
import re
import cv2
import math
import copy
import shutil
import random
import numpy as np
from tqdm import tqdm
from cv2 import dnn_superres
from PIL import Image, ImageOps
from utils import load_json, save_json


def main():
    """
    train: 7174, valid: 819
    """
    true_train_files = sorted(os.listdir('handa_sharpen/train/label_proc'))
    assert true_train_files == sorted(os.listdir('handa_sharpen/train/noise_proc'))
    true_test_files = sorted(os.listdir('handa_sharpen/valid/label_proc'))
    assert true_test_files == sorted(os.listdir('handa_sharpen/valid/noise_proc'))

    true_test_files = set(true_test_files)
    true_train_files = set(true_train_files)

    models = [None, None]
    target_res = 256
    for scale in range(2, 5):
        sr_model = dnn_superres.DnnSuperResImpl_create()
        sr_model.readModel(f'pretrain_models/ESPCN_x{scale}.pb')
        sr_model.setModel('espcn', scale)
        models.append(sr_model)
    additional = 0
    for folder in ['handa_sharpen2/valid/label', 'handa_sharpen2/train/label',
                   'handa_sharpen2/valid/noise', 'handa_sharpen2/train/noise']:
        target_folder = folder + '_proc'
        os.makedirs(target_folder, exist_ok=True)
        images = sorted(os.listdir(folder))
        for image in tqdm(images, desc=folder):
            target_file = os.path.join(target_folder, image)
            if image in true_train_files:
                if 'label' in folder:
                    shutil.move(f'handa_sharpen/train/label_proc/{image}', target_file)
                else:
                    shutil.move(f'handa_sharpen/train/noise_proc/{image}', target_file)
                continue
            if image in true_test_files:
                if 'label' in folder:
                    shutil.move(f'handa_sharpen/valid/label_proc/{image}', target_file)
                else:
                    shutil.move(f'handa_sharpen/valid/noise_proc/{image}', target_file)
                continue
            additional += 1
            img = cv2.imread(os.path.join(folder, image), 0)
            if 'label' in folder:
                # _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1)
                # img = cv2.medianBlur(img, 5)  # 中值滤波

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            height, width, _ = img.shape
            while height < target_res and width < target_res:
                max_ratio = max(math.ceil(target_res / height), math.ceil(target_res / width))
                if max_ratio >= 4:
                    img = models[4].upsample(img)
                elif max_ratio == 3:
                    img = models[3].upsample(img)
                else:
                    img = models[2].upsample(img)
                height, width, _ = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if 'label' in folder:
                _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
            cv2.imwrite(target_file, img)
    print(additional)


def get_complete_data():
    os.makedirs('../hanzi_filter/handa/noise_images', exist_ok=True)
    os.makedirs('../hanzi_filter/handa/output_images', exist_ok=True)
    for path in ['data_filter_tra_train', 'data_filter_tra_test', 'log_case_test_52_data',
                 'data_filter_tra_train_com', 'data_filter_tra_test_com']:
        meta = load_json(f'../hanzi_filter/handa/{path}.json')
        is_tuple = isinstance(meta[0], list)
        for book in tqdm(meta, desc=path):
            if is_tuple:
                book = book[0]
            new_chars = []
            for ch, img in book['characters']:
                image_base = os.path.basename(img)
                shutil.copy(f'../hanzi_filter/{img}', '../hanzi_filter/handa/noise_images')
                new_chars.append((ch, f'handa/output_images/{image_base}'))
            book['characters'] = new_chars
        save_json(meta, f'../hanzi_filter/handa/{path}_clean.json')
        for book in tqdm(meta, desc=path):
            if is_tuple:
                book = book[0]
            new_chars = []
            for ch, img in book['characters']:
                image_base = os.path.basename(img)
                shutil.copy(f'../hanzi_filter/{img}', '../hanzi_filter/handa/noise_images')
                new_chars.append((ch, f'handa/output_unet_images/{image_base}'))
            book['characters'] = new_chars
        save_json(meta, f'../hanzi_filter/handa/{path}_clean_unet.json')


def output_threshold():
    """
    将生成的图片做二值化
    """
    src_path = ['handa_sharpen/valid/output', 'handa_sharpen/valid/output_scale']
    target_path = ['handa_sharpen/valid/output_bin', 'handa_sharpen/valid/output_scale_bin']
    for src, target in zip(src_path, target_path):
        os.makedirs(target, exist_ok=True)
        for image in tqdm(sorted(os.listdir(src)), desc=src):
            img = cv2.imread(os.path.join(src, image), 0)
            _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1)
            cv2.imwrite(os.path.join(target, image), img)


def get_img_arr(img: Image) -> np.array:
    w, h = img.size
    arr = np.array(img.getdata()).reshape(h, w)
    return arr


def get_binary_arr(img: Image, bin_threshold=96) -> np.array:
    arr = get_img_arr(img)
    arr[arr >= bin_threshold] = 255
    arr[arr < bin_threshold] = 0
    return arr


def set_img_arr(img: Image, arr: np.array):
    img.putdata(arr.flatten())


def process_transcript(img: Image) -> Image:
    img = ImageOps.invert(img)
    arr = get_binary_arr(img)
    set_img_arr(img, arr)

    # Remove padding
    bbox = img.getbbox()
    img = img.crop(bbox)
    return img


def label_inverse():
    src = ['handa_sharpen/train/label', 'handa_sharpen/valid/label']
    dst = ['handa_sharpen/train/label_inv', 'handa_sharpen/valid/label_inv']
    for src_pth, dst_pth in zip(src, dst):
        os.makedirs(dst_pth, exist_ok=True)
        for image in tqdm(sorted(os.listdir(src_pth)), desc=src_pth):
            img = Image.open(os.path.join(src_pth, image)).convert('L')
            img = process_transcript(img)
            img.save(os.path.join(dst_pth, image))


def gen_real_complete_data(file_input: str, answer_output: str, file_output: str):
    train_data = load_json('../hanzi_filter/handa/data_filter_tra_train_new.json')
    train_set = set()
    for book in train_data:
        train_set.add((book['book_name'], book['row_order']))
    with open(file_input, encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 0 and line[0].isalpha()]
    lines = list(set(lines))
    print(len(lines))  # 883
    part_to_data = {}
    inter_set = []
    for line in tqdm(lines):
        part = line[0]
        # shutil.copy(f'../hanzi_filter/handa/{part}/characters/{line}',
        #             'handa_sharpen/test/noise')
        tokens = line[:-4].split('-')
        assert len(tokens) == 3
        if (tokens[0], int(tokens[1])) in train_set:
            inter_set.append(line)
            continue
        if part not in part_to_data:
            part_to_data[part] = []
        part_to_data[part].append(line)
    # 313 570
    print(len(inter_set), sum(len(val) for val in part_to_data.values()))
    exit()
    all_data = []
    fout = open(answer_output, 'w', encoding='utf-8')
    for part, images in part_to_data.items():
        meta = load_json(f'../hanzi_filter/handa/{part}/oracle_meta_{part}.json')
        candidate_set = {}
        total_num = 0
        for image in images:
            assert image[-4:] == '.png'
            book_name, order, _ = image[:-4].split('-')
            order = int(order)
            if (book_name, order) not in candidate_set:
                candidate_set[(book_name, order)] = []
            candidate_set[(book_name, order)].append(image)
            total_num += 1
        assert total_num == len(part_to_data[part])
        found = 0
        for book in meta:
            key = (book['book_name'], book['row_order'])
            if key in candidate_set:
                assert len(candidate_set[key]) > 0
                assert isinstance(candidate_set[key][-1], str)
                candidate_set[key].append(book)
                found += 1
        assert found == len(candidate_set)
        for (book_name, order), val in candidate_set.items():
            images, book = val[:-1], val[-1]
            assert isinstance(book, dict)
            chars, mask_ids = [], []
            to_test = 0
            for cid, char in enumerate(book['r_chars']):
                if char['img'] in images and char['char'] != '■':
                    # filter those characters without answer ■
                    to_test += 1
                    mask_ids.append(cid)
                    fout.write('\t'.join([book_name, str(order), char['img'], char['char']]) + '\n')
                chars.append((char['char'], f'handa/{part}/characters/{char["img"]}'))
            if to_test != len(images):
                print(book, images)
            assert to_test == len(images)
            for mid in mask_ids:
                all_data.append(({
                    'book_name': book_name,
                    'row_order': order,
                    'characters': chars,
                }, mid))
    fout.close()
    print(len(all_data))
    save_json(all_data, file_output)


def test_log_to_data(path: str, batch_size=4):
    test_data = load_json('../hanzi_filter/handa/data_filter_tra_test.json')
    with open(path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    data_mask, cur_pos_batch, cur_idx = [], [], 0
    for line in lines:
        positions = re.findall(r'\(([0-9]+), ([0-9]+)\)', line)
        cur_pos_batch += [(int(t[0]), int(t[1]) - 1) for t in positions]
        if len(cur_pos_batch) >= batch_size:
            assert len(cur_pos_batch) == batch_size
            cur_pos_batch.sort()
            for sid, cid in cur_pos_batch:
                data_mask.append((test_data[cur_idx + sid], cid))
            cur_pos_batch = []
            cur_idx += batch_size
    if len(cur_pos_batch) > 0:
        assert len(test_data) == cur_idx + len(cur_pos_batch)
        cur_pos_batch.sort()
        for sid, cid in cur_pos_batch:
            data_mask.append((test_data[cur_idx + sid], cid))
    output_path = path.replace('.txt', '_data.json')
    save_json(data_mask, output_path)


def check_data():
    print(len(load_json('../hanzi_filter/handa/data_filter_tra_train.json')))
    print(len(load_json('../hanzi_filter/handa/data_filter_tra_test.json')))
    print(len(load_json('../hanzi_filter/handa/data_filter_tra_train_com.json') +
              load_json('../hanzi_filter/handa/data_filter_tra_test_com.json')))
    exit()
    fout = open('../hanzi_filter/handa/book_order.txt', 'w', encoding='utf-8')
    data = load_json('../hanzi_filter/handa/data_filter_tra_train.json')
    for book in data:
        text = ''.join(ch for ch, _ in book['characters'])
        fout.write(book['book_name'] + '\t' + str(book['row_order']) + '\t' + text + '\n')
    fout.close()


def find_test_vague():
    random.seed(100)
    sample_data = []
    test_dir = '../hanzi_filter/handa/log_case_test_52_data.json'
    test_data = load_json(test_dir)
    threshold = 0.9
    for book, _ in tqdm(test_data, 'test'):
        characters = book['characters']
        ratio_pass = []
        for char, img in characters:
            img = cv2.imread(os.path.join('../hanzi_filter', img))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ratio = np.sum(img < 100) / (img.shape[0] * img.shape[1])
            ratio_pass.append(ratio > threshold)
        if len(characters) < 3:
            assert len(characters) > 0
            if sum(ratio_pass) == len(characters):
                sample_data.append((book, random.randint(0, len(characters) - 1)))
        else:
            candidates = []
            for k in range(1, len(characters) - 1):
                if ratio_pass[k-1] and ratio_pass[k] and ratio_pass[k+1]:
                    candidates.append(k)
            if len(candidates) > 0:
                sample_data.append((book, random.choice(candidates)))
    print(len(sample_data))
    save_json(sample_data, '../hanzi_filter/handa/log_case_test_52_hard100.json')


def generate_new_train_set(file_input: str):
    train_data = load_json('../hanzi_filter/handa/data_filter_tra_train.json')
    print(len(train_data))  # 24648
    with open(file_input, encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 0 and line[0].isalpha()]
    lines = list(set(lines))
    print(len(lines))
    new_train_set, exist_train_set = [], set()
    for line in tqdm(lines):
        tokens = line[:-4].split('-')
        assert len(tokens) == 3
        exist_train_set.add((tokens[0], int(tokens[1])))
    for book in train_data:
        if (book['book_name'], book['row_order']) not in exist_train_set:
            new_train_set.append(book)
    print(len(new_train_set))  # 24182
    save_json(new_train_set, '../hanzi_filter/handa/data_filter_tra_train_new.json')


def generate_complete_mid():
    data = []
    for f in ['handa/data_filter_tra_train_com.json', 'handa/data_filter_tra_test_com.json']:
        for book in load_json(f'../hanzi_filter/{f}'):
            for cid, (ch, _) in enumerate(book['characters']):
                if ch == '■':
                    data.append((book, cid))
    save_json(data, '../hanzi_filter/handa/data_filter_tra_all_com_mid.json')


def data_stat(data):
    count, chs, books = 0, set(), set()
    for book in data:
        if isinstance(book, list) or isinstance(book, tuple):
            book = book[0]
        for ch, img in book['characters']:
            if not img.startswith('handa/extra'):
                chs.add(ch)
                count += 1
        books.add((book['book_name'], book['row_order']))
    print('stat:', count, len(chs), len(books))


def generate_test_set():
    data1 = load_json('../hanzi_filter/handa/cases_com_tra_mid_new.json')
    data2 = load_json('../hanzi_filter/handa/cases_com_tra_mid.json')
    print(len(data1), len(data2))
    set1, set2 = set(), set()
    all_data = []
    for book, mid in data1:
        set1.add((book['book_name'], book['row_order'], mid))
        all_data.append((book, mid))
    for book, mid in data2:
        if (book['book_name'], book['row_order'], mid) not in set1:
            set2.add((book['book_name'], book['row_order'], mid))
            all_data.append((book, mid))
    assert len(set1 & set2) == 0
    print(len(all_data))  # 1648
    data_stat(all_data)   # 13213 640 1032
    save_json(all_data, '../hanzi_filter/handa/cases_com_tra_mid_combine.json')


def check_confusing_characters(path: str):
    with open(path, encoding='utf-8') as fin:
        lines = fin.readlines()
    count = {
        '七': {}, '甲': {},
        '月': {}, '夕': {},
        '田': {}, '㘡': {},
    }
    pairs = {
        '七': '甲', '甲': '七',
        '月': '夕', '夕': '月',
        '田': '㘡', '㘡': '田',
    }
    correct_ans, wrong_ans = 0, 0
    for line in lines:
        line = line.strip()
        res = re.findall(r'([^\t]+)\t\(\d+, \d+\)', line)
        if len(res) > 0:
            assert len(res) == 1
            correct_ans += 1
        for lab in res:
            if lab in count:
                target = count[lab]
                target.setdefault('ori', 0)
                target['ori'] += 1
        res = re.findall(r'\'([^ ]+) ([^ ]+) \(\d+, \d+\)\'', line)
        for lab, pre in res:
            wrong_ans += 1
            if lab in count:
                target = count[lab]
                key = 'pre' if pre == pairs[lab] else 'oth'
                target.setdefault(key, 0)
                target[key] += 1
    print(correct_ans, wrong_ans)
    for key, val in count.items():
        val.setdefault('ori', 0)
        val.setdefault('pre', 0)
        val.setdefault('oth', 0)
        print(key, val['ori'], val['pre'], val['oth'])


def test_char_context(data: list, ch: str):
    pre_count, suf_count, count = {}, {}, 0
    for book in data:
        if isinstance(book, list) or isinstance(book, tuple):
            book = book[0]
        chars = book['characters']
        for i, (cur_ch, _) in enumerate(chars):
            if cur_ch == ch:
                count += 1
                key = (chars[i-1][0] if i != 0 else '^') + chars[i][0]
                pre_count.setdefault(key, 0)
                pre_count[key] += 1
                key = chars[i][0] + (chars[i+1][0] if i != len(chars) - 1 else '$')
                suf_count.setdefault(key, 0)
                suf_count[key] += 1
    return pre_count, suf_count, count


def find_bad_case(path: str):
    with open(path, encoding='utf-8') as fin:
        lines = fin.readlines()
    cur_wrong_chs, cur_topk_20 = [], []
    sample_idx = 0
    for line in lines:
        line = line.strip()
        if line.startswith('-------------'):
            assert len(cur_topk_20) == 2
            for correct, wrong, idx in cur_wrong_chs:
                idx = int(idx)
                topk_20 = cur_topk_20[idx]
                exist = False
                for ch, rate in topk_20:
                    if ch == correct:
                        exist = True
                        break
                if not exist:
                    yield sample_idx + idx
            sample_idx += len(cur_topk_20)
            cur_wrong_chs, cur_topk_20 = [], []
            continue
        elif line.startswith('Wrong:'):
            cur_wrong_chs = re.findall(r'\'([^ ]+) ([^ ]+) \((\d+), \d+\)\'', line)
        elif line.startswith('Top20:'):
            topk_20 = re.findall(r'\'([^ ]+)\', ([0-1]\.\d+)', line)
            assert len(topk_20) == 20
            cur_topk_20.append(topk_20)


def round_up(value):
    if not isinstance(value, np.ndarray):
        return int(value + 0.5)
    else:
        return (value + 0.5).astype(int)


def extract_rotated_rectangle(image: np.ndarray, x: float, y: float, w: float, h: float, rot: float) -> np.ndarray:
    """
    image: cv2.imread 得到的图片矩阵
    x, y, w, h, rot: 对应 x, y, width, height, rotation
    """
    height, width, n_chan = image.shape
    x = width * x / 100
    y = height * y / 100
    w = width * w / 100
    h = height * h / 100
    rot = rot * np.pi / 180  # 弧度制，顺时针旋转角
    # vertexes
    cnt = np.array([
        [[x, y]],
        [[x - h * np.sin(rot), y + h * np.cos(rot)]],
        [[x - h * np.sin(rot) + w * np.cos(rot), y + h * np.cos(rot) + w * np.sin(rot)]],
        [[x + w * np.cos(rot), y + w * np.sin(rot)]],
    ])
    rect = cv2.minAreaRect(round_up(cnt))

    # the order of the box points: bottom left, top left, top right, bottom right
    box = round_up(cv2.boxPoints(rect))
    # adjust the order
    if abs(box[0][0] - box[1][0]) > abs(box[0][0] - box[3][0]):
        temp_box = box.copy()
        box[0, :] = temp_box[3, :]
        box[1:4, :] = temp_box[0:3, :]
        rec_h = round_up(rect[1][0])
        rec_w = round_up(rect[1][1])
    else:
        # get width and height of the detected rectangle
        rec_w = round_up(rect[1][0])
        rec_h = round_up(rect[1][1])
    # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    src_pts = box.astype("float32")
    assert src_pts.shape == (4, 2)
    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([
        [0, rec_h - 1],
        [0, 0],
        [rec_w - 1, 0],
        [rec_w - 1, rec_h - 1],
    ])
    dst_pts = dst_pts.astype("float32")
    # the perspective transformation matrix
    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    wrapped = cv2.warpPerspective(image, mat, (rec_w, rec_h))
    return wrapped


def generate_case_file():
    path_base = 'handa/2022-06-21_DEMO_H00137_to_SongChenyang'
    data = load_json('/'.join([path_base, 'H00137annotation.json']))[0]['annotations'][0]['result']
    data = [item['value'] for item in data]
    lab_data = [item['labels'][0].startswith('Damaged_inscription') for item in data if 'labels' in item]
    txt_data = [item for item in data if 'text' in item]
    # 104 104 13
    print(len(lab_data), len(txt_data), sum(lab_data), [item['text'][0] for item in txt_data])
    # ['小', '卯', '芻', '𦎫', '丑', '卜', '㞢', '𡆥', '允', '\U000fe033', '尿', '于', '彔']
    print([txt['text'][0] for txt, lab in zip(txt_data, lab_data) if lab])
    lengths = [2, 1, 1, 1, 1, 1, 6, 26, 33, 32]
    assert sum(lengths) == len(lab_data) == len(txt_data)
    image = cv2.imread('/'.join([path_base, 'H00137zheng.jpg']))
    print(image.shape)  # (3658, 2336, 3)
    results_see, results_com = [], []
    accu_len = 0
    for lid, le in enumerate(lengths):
        cur_sent, targets = [], []
        for idx in range(accu_len, accu_len + le):
            x, y, w, h, rot = txt_data[idx]['x'], txt_data[idx]['y'], \
                txt_data[idx]['width'], txt_data[idx]['height'], txt_data[idx]['rotation']
            img_name = '/'.join([path_base, 'images', f'{lid}-{idx}.png'])
            wrapped = extract_rotated_rectangle(image, x, y, w, h, rot)
            cv2.imwrite(img_name, wrapped)
            cur_sent.append([txt_data[idx]['text'][0], img_name])
            if lab_data[idx]:
                targets.append(idx - accu_len)
        accu_len += le
        print(''.join([item[0] for item in cur_sent]))
        if len(targets) == 0:
            continue
        mask_chars = copy.deepcopy(cur_sent)
        for target in targets:
            chars = copy.deepcopy(cur_sent)
            results_see.append(({
                'book_name': 'H00137正',
                'row_order': lid + 1,
                'characters': chars,
            }, target))
            mask_chars[target][0] = '■'
        for target in targets:
            results_com.append(({
                'book_name': 'H00137正',
                'row_order': lid + 1,
                'characters': mask_chars,
            }, target))
    save_json(results_see, 'handa/cases_H00137zheng_see.json')
    save_json(results_com, 'handa/cases_H00137zheng_com.json')


def generate_new_embedding():
    import torch
    import opencc
    import torch.nn as nn

    old_char_to_id = load_json('../chinese-bert-wwm-ext/vocab_old.json')
    converter = opencc.OpenCC('t2s.json')
    old_vocab_size, new_vocab_size, hidden_size, pad_token_id = 21128, 4116, 768, 0
    assert old_vocab_size == len(old_char_to_id)
    with open('../chinese-bert-wwm-ext/vocab.txt', 'r', encoding='utf-8') as fin:
        exist_vocab_list = [line[:-1] for line in fin.readlines() if len(line) > 1]
    assert len(exist_vocab_list) == new_vocab_size

    main_keys = {
        'bert.embeddings.word_embeddings.weight': nn.Embedding(
            new_vocab_size, hidden_size, padding_idx=pad_token_id).weight.detach(),
        'cls.predictions.bias': torch.zeros(new_vocab_size),
        'cls.predictions.decoder.weight': nn.Linear(hidden_size, new_vocab_size, bias=False).weight.detach(),
    }
    state_dict = torch.load('../chinese-bert-wwm-ext/pytorch_model_old.bin')

    for idx, char in tqdm(enumerate(exist_vocab_list)):
        update_cnt = -1
        for key, val in main_keys.items():
            if val.ndim == 2:
                new_vec = torch.zeros(hidden_size, dtype=val.dtype)
            else:
                new_vec = torch.tensor(0, dtype=val.dtype)
            cur_cnt = 0
            if idx >= 113:
                for ch in char:
                    if ch in old_char_to_id:
                        new_vec += state_dict[key][old_char_to_id[ch]]
                        cur_cnt += 1
                for ch in converter.convert(char):
                    if ch in old_char_to_id:
                        new_vec += state_dict[key][old_char_to_id[ch]]
                        cur_cnt += 1
            else:
                if char in old_char_to_id:
                    new_vec += state_dict[key][old_char_to_id[char]]
                    cur_cnt += 1
            if cur_cnt > 0:
                val[idx] = new_vec / cur_cnt
            if update_cnt < 0:
                update_cnt = cur_cnt
            assert update_cnt == cur_cnt
    for key, val in main_keys.items():
        state_dict[key] = val
    torch.save(state_dict, '../chinese-bert-wwm-ext/pytorch_model.bin')


if __name__ == '__main__':
    # find_test_vague()
    # check_data()
    # exit()
    # test_log_to_data('output/finetune_single_mlm_np_neo/log_case_test_52.txt')
    # main()
    # get_complete_data()
    # label_inverse()
    # find_bad_case('output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new/log_case_test_45_1648_cross_mk25.txt')
    generate_new_embedding()
    # generate_case_file()
    exit()
    check_confusing_characters('output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new'
                               '/log_case_test_45_1648_cross_mk25.txt')
    check_confusing_characters('output/tra_finetune_single_mlm_p1_load_image_mk50_unrec_uncls_new'
                               '/log_case_test_57_1648_image.txt')
    check_confusing_characters('output/tra_finetune_single_mlm_p0_text_all_new'
                               '/log_case_test_20_1648.txt')
    gen_real_complete_data('handa/据图及上下文推断top20字.txt', 'handa/answer.txt',
                           '../hanzi_filter/handa/cases_com_tra_mid.json')
    # gen_real_complete_data('handa/2022-05-21_res_char_to_songchenyang.txt', 'handa/answer_new.txt',
    #                        '../hanzi_filter/handa/cases_com_tra_mid_new.json')
    # gen_real_complete_data('handa/2022-05-21_res_char_to_songchenyang_not_perfect.txt', 'handa/answer_new_not.txt',
    #                        '../hanzi_filter/handa/cases_com_tra_mid_new_not_perfect.json')
    # generate_new_train_set('handa/2022-05-21_res_char_to_songchenyang.txt')
    # output_threshold()
    # generate_complete_mid()
    generate_test_set()
