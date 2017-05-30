""" Library for image processing """
import os
import logging
import hashlib
import io
import urllib.request
from PIL import Image
import numpy as np

LOGGER = logging.getLogger('imageutils')

def img_to_array(img, data_format='channels_last', data_type=np.float32):
    """Converts a PIL Image instance to a Numpy array."""
    if data_format not in {'channels_first', 'channels_last'}:
        LOGGER.warning('Unknown/missing data format in img_to_array function: ' + data_format)
        return None
        #raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    # 'channels_first' by default for tensorflow image format
    img_data = np.asarray(img, dtype=data_type)
    if len(img_data.shape) == 3:
        if data_format == 'channels_first':
            img_data = img_data.transpose(2, 0, 1)
    elif len(img_data.shape) == 2:
        if data_format == 'channels_first':
            img_data = img_data.reshape((1, img_data.shape[0], img_data.shape[1]))
        else:
            img_data = img_data.reshape((img_data.shape[0], img_data.shape[1], 1))
    else:
        LOGGER.warning('Unsupported image shape in img_to_array function: ' + str(img_data.shape))
        return None
        #raise ValueError('Unsupported image shape: ', img_data.shape)
    return img_data


def resize_img(img, target_size):
    """Return a resized PIL image."""
    if target_size is None \
       or not isinstance(target_size, (list, tuple, np.ndarray)) \
       or len(target_size) != 2:
        LOGGER.warning('Unexpected size dimension for resize_img function: ' + str(target_size))
        return None
    hw_tuple = (target_size[1], target_size[0])
    if img.size != hw_tuple:
        img = img.resize(hw_tuple, Image.ANTIALIAS)
    return img


def crop_img(img, target_crop):
    """Returns a cropped PIL image."""
    if target_crop is None \
       or not isinstance(target_crop, (list, tuple, np.ndarray)) \
       or len(target_crop) != 4:
        LOGGER.warning('Unexpected crop dimension for crop_img function: ' + str(target_crop))
        return None
    img = img.crop(target_crop)
    return img


def load_img(path, target_size=None):
    """Loads an image into PIL format."""
    if Image is None:
        LOGGER.warning('Could not import PIL.Image. The use of `load_img` requires PIL.')
        return None
        #raise ImportError('Could not import PIL.Image. '
        #                  'The use of `load_img` requires PIL.')
    if not os.path.exists(path):
        LOGGER.warning('Image not found in load_img function: ' + path)
        return None
        #raise ValueError('Directory not found: ', path)
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if target_size:
        img = resize_img(img, target_size=target_size)
    return img


def save_img(img, output_path, quality=95):
    """Saves PIL image with a provided filename."""
    if not output_path or not os.path.exists(os.path.dirname(output_path)):
        LOGGER.warning('Directory not found in save_img function: ' + output_path)
        #raise ValueError('Directory not found: ', os.path.dirname(output_path))
    else:
        img.save(output_path, 'jpeg', quality=quality)


def load_img_from_md5(image_md5, url='https://octopus.heuritech.com/get?md5=', target_size=None):
    """Load PIL image by getting image md5 from url (octopus by default)."""
    try:
        md5_url = url + image_md5
        file_data = io.BytesIO(urllib.request.urlopen(md5_url).read())
        img = Image.open(file_data)
        if target_size:
            img = resize_img(img, target_size=target_size)
    except ValueError as error:
        LOGGER.warning('Error in load_img_from_md5: ' + str(error))
        return None
    return img


def get_md5_from_imagepath(image_path):
    """Computes and returns md5 from image (providing its path)."""
    if image_path is None or not os.path.exists(image_path):
        LOGGER.warning('Image not found in get_md5_from_imagepath function: ' + image_path)
        return None
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as file_data:
        for chunk in iter(lambda: file_data.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
