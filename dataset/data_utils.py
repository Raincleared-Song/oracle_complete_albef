import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms


def random_transform(image: Image.Image, do_trans: bool, pad_color=0, mask_ratio=0.0, noise_ratio=0.25) -> np.ndarray:
    """
    Image, cv2 <-> array: (height, width, dim)
    随机变换：[0, erase * 4] * [0, rot90, rot180, rot270] (在外面做) * [0, 25% noise, 50% noise, 75% noise]
    """
    image = np.array(image)
    if not do_trans:
        return image
    # 覆盖变换
    image = random_mask(image, pad_color=pad_color, mask_ratio=mask_ratio)
    # 噪点变换
    if noise_ratio > 0:
        # 按比例加噪声
        image = add_random_noise(image, ratio=noise_ratio, noise_color=(255 - pad_color))
    else:
        image = random_noise(image, noise_color=(255 - pad_color))
    return image


def random_mask(image: np.ndarray, pad_color=0, mask_ratio=0.0) -> np.ndarray:
    mask_image = image.copy()
    height, width = mask_image.shape[:2]
    rnd = random.random()
    mask_probs = [0.5, 0.625, 0.75, 0.875]
    if rnd < mask_probs[0]:
        # 不做覆盖
        pass
    elif rnd < mask_probs[1]:
        # 上 -> 下覆盖
        mask_height = int(height * mask_ratio)
        mask_image[:mask_height, :] = pad_color
    elif rnd < mask_probs[2]:
        # 下 -> 上覆盖
        mask_height = height - int(height * mask_ratio)
        mask_image[mask_height:, :] = pad_color
    elif rnd < mask_probs[3]:
        # 左 -> 右覆盖
        mask_width = int(width * mask_ratio)
        mask_image[:, :mask_width] = pad_color
    else:
        # 右 -> 左覆盖
        mask_width = width - int(width * mask_ratio)
        mask_image[:, mask_width:] = pad_color
    return mask_image


def random_rotate(image: np.ndarray) -> np.ndarray:
    rot_image = image.copy()
    rnd = random.random()
    rot_probs = [0.7, 0.8, 0.9]
    if rnd < rot_probs[0]:
        # 不做旋转
        pass
    elif rnd < rot_probs[1]:
        # 旋转 180 度
        rot_image = cv2.flip(rot_image, -1)
    else:
        rot_image = cv2.transpose(rot_image)
        if rnd < rot_probs[2]:
            # 顺时针旋转 90 度
            rot_image = cv2.flip(rot_image, 1)
        else:
            # 逆时针旋转 90 度
            rot_image = cv2.flip(rot_image, 0)
    return rot_image


def random_noise(image: np.ndarray, noise_color=255) -> np.ndarray:
    noise_image = image.copy()
    mask_probs = [0.4, 0.6, 0.8]
    rnd = random.random()
    if rnd < mask_probs[0]:
        # 不做操作
        pass
    elif rnd < mask_probs[1]:
        noise_image = add_random_noise(noise_image, 0.25, noise_color)
    elif rnd < mask_probs[2]:
        noise_image = add_random_noise(noise_image, 0.50, noise_color)
    else:
        noise_image = add_random_noise(noise_image, 0.75, noise_color)
    return noise_image


def add_random_noise(image: np.ndarray, ratio=0.25, noise_color=255) -> np.ndarray:
    """
    加椒盐噪声，ratio 为噪点的比例
    """
    noise_image = image.copy()
    height, width = noise_image.shape[:2]
    mask = np.random.random((height, width)) < ratio
    h_mask, w_mask = np.nonzero(mask)
    noise_image[h_mask, w_mask] = noise_color
    return noise_image


def resize_pad_image(image: Image.Image, shape: tuple, do_trans: bool,
                     pad_color=0, mask_ratio=0.0, noise_ratio=0.25, do_rotate=False) -> np.ndarray:
    # resize
    width, height = image.size
    assert len(shape) == 2
    r_width, r_height = shape
    w_ratio, h_ratio = r_width / width, r_height / height
    if w_ratio >= h_ratio:
        # resize by height
        image = image.resize(size=(int(width * h_ratio), r_height))
    else:
        # resize by width
        image = image.resize(size=(r_width, int(height * w_ratio)))
    width, height = image.size
    pad_h, pad_w = (r_height - height) // 2, (r_width - width) // 2
    if image.mode == 'L':
        pad_image = np.full((r_height, r_width), pad_color, dtype=np.uint8)
        pad_image[pad_h:pad_h + height, pad_w:pad_w + width] = random_transform(
            image, do_trans, pad_color=pad_color, mask_ratio=mask_ratio, noise_ratio=noise_ratio)
    else:
        # padding to shape, dtype required
        pad_image = np.full((r_height, r_width, 3), pad_color, dtype=np.uint8)
        pad_image[pad_h:pad_h+height, pad_w:pad_w+width, :] = random_transform(
            image, do_trans, pad_color=pad_color, mask_ratio=mask_ratio, noise_ratio=noise_ratio)
    if do_trans and do_rotate:
        pad_image = random_rotate(pad_image)
    return pad_image


def reverse_normalize(mean: list, std: list) -> transforms.Compose:
    rev_mean = [-m for m in mean]
    rev_std = [1 / s for s in std]
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=rev_std),
        transforms.Normalize(mean=rev_mean, std=[1., 1., 1.]),
    ])


def tensor_to_img(img: torch.Tensor, shape: tuple, mean: list, std: list) -> Image.Image:
    assert len(shape) == 3 and shape[0] in [1, 3]
    img = img.view(shape)
    rev_trans = reverse_normalize(mean, std)
    img = rev_trans(img) * 255
    img = np.moveaxis(img.numpy(), 0, -1).astype(np.uint8)
    return Image.fromarray(img, mode=('L' if shape[0] == 1 else 'RGB'))
