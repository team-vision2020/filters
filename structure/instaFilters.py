# Compiled by Chunlok
# Ported by Joel on 10/26 with batch work

import numpy as np
import skimage
from skimage import io, filters

def channel_adjust(channel, values):
    orig_size = channel.shape
    flat_channel = channel.flatten()
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)
    return adjusted.reshape(orig_size)

def split_images_into_channels(images):
    """Look at each image separately"""
    red_channel = images[... , 0]
    green_channel = images[... , 1]
    blue_channel = images[... , 2]
    return red_channel, green_channel, blue_channel

def merge_channels(red, green, blue):
    """Merge channels back into an image"""
    return np.stack([red, green, blue], axis=-1)

def sharpen(image, a, b):
    """Sharpening an image: Blur and then subtract from original"""
    blurred = skimage.filters.gaussian_filter(image, sigma=10, multichannel=True)
    sharper = np.clip(image * a - blurred * b, 0, 1.0)
    return sharper

# Note channel(s) in here refer to the same channel on multiple images (c x flat_img
def channels_adjust(channels, values):
    orig_size = channels.shape
    flat_channels = channels.reshape(channels.shape[0], -1) # flat channel info per image
    # lol we'll figure something out here
    adjusted = np.asarray([np.interp(img_channel, np.linspace(0, 1, len(values)), values) for img_channel in flat_channels])
#     adjusted = multiInterp(flat_channels, np.linspace(0, 1, len(values)), values)
    return adjusted.reshape(orig_size)

# From SO: https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array
def multiInterp(x, xp, fp):
    i = np.arange(x.size)
    j = np.searchsorted(xp, x) - 1
    d = (x - xp[j]) / (xp[j + 1] - xp[j])
    return (1 - d) * fp[i, j] + fp[i, j + 1] * d


def contrast_adjust(images, values):
    r, g, b = split_images_into_channels(images)
    r = channel_adjust(r, values)
    g = channel_adjust(g, values)
    b = channel_adjust(b, values)
    adjusted_im = merge_channels(r, g, b)
    return adjusted_im

from skimage import color
def change_saturation(image, offset):
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * offset, 0, 1)
    return color.hsv2rgb(hsv)

# todo: vectorize?
def change_saturations(images, offset):
    return np.asarray([change_saturation(image, offset) for image in images])

def gotham_single(image):
    # too hard to parse, going to inefficient this
    print(image.shape)
    original_image = skimage.img_as_float(image)
    r = original_image[:, :, 0]
    b = original_image[:, :, 2]
    r_boost_lower = channel_adjust(r, [
        0, 0.05, 0.1, 0.2, 0.3,
        0.5, 0.7, 0.8, 0.9,
        0.95, 1.0])
    b_more = np.clip(b + 0.03, 0, 1.0)
    merged = np.stack([r_boost_lower, original_image[:, :, 1], b_more], axis=2)
    blurred = filters.gaussian(merged, sigma=10, multichannel=True)
    final = np.clip(merged * 1.3 - blurred * 0.3, 0, 1.0)
    b = final[:, :, 2]
    b_adjusted = channel_adjust(b, [
        0, 0.047, 0.118, 0.251, 0.318,
        0.392, 0.42, 0.439, 0.475,
        0.561, 0.58, 0.627, 0.671,
        0.733, 0.847, 0.925, 1])
    final[:, :, 2] = b_adjusted
    return final

def gotham(images):
    print(images.shape)
    if images.ndim == 3:
        return gotham_single(images)
    return np.asarray([gotham_single(images[i]) for i in range(images.shape[0])])

def clarendon(images):
    #im = change_saturation(im, 1.5)
    #im = contrast_adjust(im, [0, 0.1, 0.5, 0.7, 0.9, 1])
    r, g, b = split_images_into_channels(images)
    r = channels_adjust(r, [0, 0.20, 0.5, 0.9, 1])
    g = channels_adjust(g, [0, 0.25, 0.6, 0.8, 1])
    b = channels_adjust(b, [0, 0.3, 0.55, 0.8, 1])
    im = merge_channels(r, g, b)
    return im

def gingham(images):
    #im = change_saturation(im, 1.5)
    #im = contrast_adjust(im, [0, 0.1, 0.5, 0.7, 0.9, 1])
    r, g, b = split_images_into_channels(images)
    r = channels_adjust(r, [0.2, 0.4, 0.7, 0.85, 1])
    g = channels_adjust(g, [0.2, 0.35, 0.6, 0.8, .9])
    b = channels_adjust(b, [0.2, 0.35, 0.6, 0.8, .9])
    im = merge_channels(r, g, b)
    return im

def juno(images):
    images = contrast_adjust(images, [0, 0.15, 0.5, 0.85, 1])
    r, g, b = split_images_into_channels(images)
    #r = channel_adjust(r, [0.2, 0.4, 0.7, 0.85, 1])
    g = channels_adjust(g, [0.1, 0.5, 1])
    #b = channel_adjust(b, [0.2, 0.35, 0.6, 0.8, .9])
    im = merge_channels(r, g, b)
    return im

def lark(images):
    images = contrast_adjust(images, [0, 0.2, 0.5, 0.85, 1])
    r, g, b = split_images_into_channels(images)
    #r = channel_adjust(r, [0.2, 0.4, 0.7, 0.85, 1])
    g = channels_adjust(g, [0.1, 0.28, 0.55, 0.8, 1])
    #b = channel_adjust(b, [0.2, 0.35, 0.6, 0.8, .9])
    im = merge_channels(r, g, b)
    return im

def reyes(images):
    images = contrast_adjust(images, [0.1, 0.5, 0.8, 0.9, 1])
    r, g, b = split_images_into_channels(images)
    r = channels_adjust(r, [0.0, 0.25, 0.55, 0.76, 1])
    g = channels_adjust(g, [0.0, 0.25, 0.50, 0.75, 0.97])
    b = channels_adjust(b, [0.0, 0.22, 0.5, 0.75, .9])
    im = merge_channels(r, g, b)
    return im
