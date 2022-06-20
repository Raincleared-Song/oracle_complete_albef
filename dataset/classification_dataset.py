import os
from PIL import Image
from utils import load_json
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.data_utils import resize_pad_image


class ImageClassificationDataset(Dataset):

    src_to_idx = {'wzb': 0, 'chant': 1}
    idx_to_dir = ['/home/linbiyuan/corpus/wenbian/labels_页数+序号+著录号+字形_校对版_060616/char',
                  '/data/private/songchenyang/hanzi_filter/handa']
    src_to_pad_color = [255, 0]

    def __init__(self, config, mode):
        file_list = config['train_file'] if mode == 'train' else config['test_file']
        self.data = []
        for file in file_list:
            part = load_json(os.path.join(config['data_prefix'], file))
            for key, val in part.items():
                self.data += val
        print(f'loaded {mode} data: {len(self.data)}')
        assert config['dataset_mode'] == 'classification'
        self.model_mode = mode
        if config['img_mode'] == 'RGB':
            mean, std = {
                128: ([0.5601, 0.5598, 0.5596], [0.4064, 0.4065, 0.4066]),
            }[config['image_res']]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            mean, std = {
                128: ([0.5599], [0.4065]),
            }[config['image_res']]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        char, config = self.data[index], self.config
        img_res = config['image_res']
        image, label, src = char['img'], char['lab'], char['src']
        src = ImageClassificationDataset.src_to_idx[src]
        image = os.path.join(ImageClassificationDataset.idx_to_dir[src], image)
        image = Image.open(image).convert(config['img_mode'])
        pad_color = ImageClassificationDataset.src_to_pad_color[src]
        image = resize_pad_image(image, (img_res, img_res), do_trans=config['img_random_transform'],
                                 pad_color=pad_color, mask_ratio=config['img_mask_ratio'],
                                 noise_ratio=config['img_noise_ratio'], do_rotate=config['img_do_rotate'])
        image = Image.fromarray(image, mode=config['img_mode'])
        image = self.transform(image).unsqueeze(0)
        return image, label
