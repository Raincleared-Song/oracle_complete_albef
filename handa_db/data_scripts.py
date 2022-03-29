import os
import cv2
import math
import shutil
from tqdm import tqdm
from cv2 import dnn_superres
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
    for path in ['data_filter_sim_train', 'data_filter_sim_test', 'log_case_test_52_data',
                 'data_filter_sim_train_com', 'data_filter_sim_test_com']:
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


if __name__ == '__main__':
    # main()
    get_complete_data()
    # output_threshold()
