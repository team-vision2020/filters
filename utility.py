""""
Utility functions.

Author: Mert Dumenci
"""

import os
import matplotlib.pyplot as plt
import skimage


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def image_from_path(path):
    return skimage.img_as_float(plt.imread(path))


def image_named(name):
    return image_from_path(os.path.join(ROOT_DIR, name))
