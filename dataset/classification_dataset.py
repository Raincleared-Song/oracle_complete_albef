import os
from PIL import Image
from utils import load_json
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.data_utils import resize_pad_image


class ImageClassificationDataset(Dataset):

    src_to_idx = {'wzb': 0, 'chant': 1}
    idx_to_dir = ['../hanzi_filter/wzb', '../hanzi_filter/handa']

    def __init__(self, config, mode):
        file_list = config['train_file'] if mode == 'train' else config['test_file']
        self.data = []
        for file in file_list:
            part = load_json(os.path.join(config['data_prefix'], file))
            for val in part:
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
        image_p, label, src = char['img'], char['lab'], char['src']
        assert 0 <= label < config['vocab_size']
        if src in ImageClassificationDataset.src_to_idx:
            src = ImageClassificationDataset.src_to_idx[src]
            image_p = os.path.join(ImageClassificationDataset.idx_to_dir[src], image_p)
        image = Image.open(image_p).convert(config['img_mode'])
        image = resize_pad_image(image, (img_res, img_res), do_trans=config['img_random_transform'],
                                 reverse=(src == 'wzb'), pad_color=0, mask_ratio=config['img_mask_ratio'],
                                 noise_ratio=config['img_noise_ratio'], do_rotate=config['img_do_rotate'])
        image = Image.fromarray(image, mode=config['img_mode'])
        image = self.transform(image).unsqueeze(0)
        return image, label, image_p
