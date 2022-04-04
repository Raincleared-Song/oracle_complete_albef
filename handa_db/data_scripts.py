import os
import re
import cv2
import math
import shutil
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


def gen_real_complete_data():
    train_data = load_json('../hanzi_filter/handa/data_filter_tra_train.json')
    train_set = set()
    for book in train_data:
        train_set.add((book['book_name'], book['row_order']))
    with open('handa/据图及上下文推断top20字.txt', encoding='utf-8') as fin:
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
    fout = open('handa/answer.txt', 'w', encoding='utf-8')
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
            chars = []
            to_test = 0
            for char in book['r_chars']:
                if char['img'] in images:
                    chars.append(('■', f'handa/{part}/characters/{char["img"]}'))
                    to_test += 1
                    fout.write('\t'.join([book_name, str(order), char['img'], char['char']]) + '\n')
                else:
                    chars.append((char['char'], f'handa/{part}/characters/{char["img"]}'))
            try:
                assert to_test == len(images)
            except AssertionError as err:
                from IPython import embed
                embed()
                raise err
            all_data.append({
                'book_name': book_name,
                'row_order': order,
                'characters': chars,
            })
    fout.close()
    save_json(all_data, '../hanzi_filter/handa/cases_com_tra.json')


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
    fout = open('../hanzi_filter/handa/book_order.txt', 'w', encoding='utf-8')
    data = load_json('../hanzi_filter/handa/data_filter_tra_train.json')
    for book in data:
        text = ''.join(ch for ch, _ in book['characters'])
        fout.write(book['book_name'] + '\t' + str(book['row_order']) + '\t' + text + '\n')
    fout.close()


if __name__ == '__main__':
    # test_log_to_data('output/finetune_single_mlm_np_neo/log_case_test_52.txt')
    # main()
    # get_complete_data()
    # label_inverse()
    gen_real_complete_data()
    # output_threshold()
