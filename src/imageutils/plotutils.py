"""utils to plot images"""

import os
import math
import matplotlib.pyplot as plt

from imageutils.imageutils import load_img

def plot_image_path(image_path, log_image_path=False):
    """plot image given image_path"""
    img = load_img(image_path)
    if log_image_path:
        plt.title(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def plot_list_image_path(list_image_path, log_image_path=False):
    """plot several images given a list of image_path in an harmonious way"""
    i = 1
    nb_img = len(list_image_path)
    plt.figure(figsize=(10, 2 * nb_img))
    for image_path in list_image_path:
        if not os.path.isfile(image_path):
            continue
        img = load_img(image_path)
        plt.subplot(math.ceil(nb_img/3) + 1, 3, i)
        i += 1
        if log_image_path:
            plt.title(image_path)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def plot_numpy_img(np_img):
    """plot image from numpy array"""
    plt.imshow(np_img, interpolation='nearest')
    plt.show()
