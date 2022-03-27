import random
import numpy as np
from PIL import Image


def random_mask(image: Image.Image, pad_color=255, mask_ratio=0.0) -> np.ndarray:
    """
    随机遮蔽，先等比例选一个遮蔽方向
    """
    image = np.array(image)
    height, width = image.shape[:2]
    if mask_ratio <= 0.0:
        return image
    rnd = random.random()
    if rnd < 0.25:
        # 上 -> 下覆盖
        mask_height = int(height * mask_ratio)
        image[:mask_height, :] = pad_color
    elif rnd < 0.5:
        # 下 -> 上覆盖
        mask_height = height - int(height * mask_ratio)
        image[mask_height:, :] = pad_color
    elif rnd < 0.75:
        # 左 -> 右覆盖
        mask_width = int(width * mask_ratio)
        image[:, :mask_width] = pad_color
    else:
        # 右 -> 左覆盖
        mask_width = width - int(width * mask_ratio)
        image[:, mask_width:] = pad_color
    return image


def resize_pad_image(image: Image.Image, shape: tuple, pad_color=255, mask_ratio=0.0) -> np.ndarray:
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
        pad_image[pad_h:pad_h + height, pad_w:pad_w + width] = random_mask(image, pad_color, mask_ratio)
    else:
        # padding to shape, dtype required
        pad_image = np.full((r_height, r_width, 3), pad_color, dtype=np.uint8)
        pad_image[pad_h:pad_h+height, pad_w:pad_w+width, :] = random_mask(image, pad_color, mask_ratio)
    return pad_image
