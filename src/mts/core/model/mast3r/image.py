# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def img_to_arr(img):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def _resize_to_multiple_of_16(img):
    """ Resize an image so both dimensions become the nearest multiple of 16
    (used when no explicit target `size` is given).
    """
    W, H = img.size
    new_W = max(16, int(round(W / 16)) * 16)
    new_H = max(16, int(round(H / 16)) * 16)
    if (new_W, new_H) != (W, H):
        img = img.resize((new_W, new_H), PIL.Image.LANCZOS)
    return img


def load_single_image(path, size=None, square_ok=False, verbose=True):
    """ Open and convert a single image to the proper input format for DUSt3R.

    If `size` is None, the image is simply resized so that both its width and
    height become the nearest multiple of 16 (no cropping is applied).
    Otherwise the original behavior is preserved: resize long/short edge to
    `size`, then center-crop to a multiple-of-16/8 box.
    """
    img = exif_transpose(PIL.Image.open(path)).convert('RGB')
    W1, H1 = img.size

    if size is None:
        # just snap to the nearest multiple of 16, no cropping
        img = _resize_to_multiple_of_16(img)
    else:
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)

        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not square_ok and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    W2, H2 = img.size
    if verbose:
        print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')

    return dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        idx=0,
        instance='0',
    )


def load_images(folder_or_list, size=None, square_ok=False, verbose=True):
    """ Open and convert all images in a list or folder to proper input format for DUSt3R.

    `size`: target size as before (224 or e.g. 512). If left as None, each
    image is simply scaled so its width/height become the nearest multiple
    of 16, instead of the fixed 224/512 resize+crop pipeline.
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        full_path = os.path.join(root, path)
        entry = load_single_image(full_path, size=size, square_ok=square_ok, verbose=verbose)
        entry['idx'] = len(imgs)
        entry['instance'] = str(len(imgs))
        imgs.append(entry)

    assert imgs, 'no images foud at ' + root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs