"""
Instagram-like filters.

Courtesy of Chunlok Lo.
"""

import numpy as np
import skimage
import skimage.filters
import skimage.color
import keras.utils


def _channel_adjust(channel, values):
    orig_size = channel.shape
    flat_channel = channel.flatten()
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)
    return adjusted.reshape(orig_size)


def _split_image_into_channels(image):
    """Look at each image separately"""
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel


def _merge_channels(red, green, blue):
    """Merge channels back into an image"""
    return np.stack([red, green, blue], axis=2)


def sharpen(image, a, b):
    """Sharpening an image: Blur and then subtract from original"""
    blurred = skimage.filters.gaussian_filter(image, sigma=10, multichannel=True)
    sharper = np.clip(image * a - blurred * b, 0, 1.0)
    return sharper


def contrast_adjust(image, values):
    r, g, b = _split_image_into_channels(image)
    r = _channel_adjust(r, values)
    g = _channel_adjust(g, values)
    b = _channel_adjust(b, values)
    adjusted_im = _merge_channels(r, g, b)
    return adjusted_im


def _change_saturation(image, offset):
    hsv = skimage.color.rgb2hsv(image)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * offset, 0, 1)
    return skimage.color.hsv2rgb(hsv)


def _gotham(image):
    original_image = skimage.img_as_float(image)
    r = original_image[:, :, 0]
    b = original_image[:, :, 2]
    r_boost_lower = _channel_adjust(r, [
        0, 0.05, 0.1, 0.2, 0.3,
        0.5, 0.7, 0.8, 0.9,
        0.95, 1.0])
    b_more = np.clip(b + 0.03, 0, 1.0)
    merged = np.stack([r_boost_lower, original_image[:, :, 1], b_more], axis=2)
    blurred = skimage.filters.gaussian(merged, sigma=10, multichannel=True)
    final = np.clip(merged * 1.3 - blurred * 0.3, 0, 1.0)
    b = final[:, :, 2]
    b_adjusted = _channel_adjust(b, [
        0, 0.047, 0.118, 0.251, 0.318,
        0.392, 0.42, 0.439, 0.475,
        0.561, 0.58, 0.627, 0.671,
        0.733, 0.847, 0.925, 1])
    final[:, :, 2] = b_adjusted
    return final


def _clarendon(image):
    original_image = skimage.img_as_float(image)
    im = original_image
    r, g, b = _split_image_into_channels(im)
    r = _channel_adjust(r, [0, 0.20, 0.5, 0.9, 1])
    g = _channel_adjust(g, [0, 0.25, 0.6, 0.8, 1])
    b = _channel_adjust(b, [0, 0.3, 0.55, 0.8, 1])
    im = _merge_channels(r, g, b)
    return im


def _gingham(image):
    original_image = skimage.img_as_float(image)
    im = original_image
    im = skimage.img_as_float(im)
    r, g, b = _split_image_into_channels(im)
    r = _channel_adjust(r, [0.2, 0.4, 0.7, 0.85, 1])
    g = _channel_adjust(g, [0.2, 0.35, 0.6, 0.8, .9])
    b = _channel_adjust(b, [0.2, 0.35, 0.6, 0.8, .9])
    im = _merge_channels(r, g, b)
    return im


def _juno(image):
    #Juno filter
    original_image = skimage.img_as_float(image)
    im = original_image
    im = contrast_adjust(im, [0, 0.15, 0.5, 0.85, 1])
    r, g, b = _split_image_into_channels(im)
    g = _channel_adjust(g, [0.1, 0.5, 1])
    im = _merge_channels(r, g, b)
    return im


def _lark(image):
    #Lark filter
    original_image = skimage.img_as_float(image)
    im = original_image
    im = contrast_adjust(im, [0, 0.2, 0.5, 0.85, 1])
    r, g, b = _split_image_into_channels(im)
    g = _channel_adjust(g, [0.1, 0.28, 0.55, 0.8, 1])
    im = _merge_channels(r, g, b)
    return im


def _reyes(image):
    #Reyes filter
    original_image = skimage.img_as_float(image)
    im = original_image
    im = contrast_adjust(im, [0.1, 0.5, 0.8, 0.9, 1])
    r, g, b = _split_image_into_channels(im)
    r = _channel_adjust(r, [0.0, 0.25, 0.55, 0.76, 1])
    g = _channel_adjust(g, [0.0, 0.25, 0.50, 0.75, 0.97])
    b = _channel_adjust(b, [0.0, 0.22, 0.5, 0.75, .9])
    im = _merge_channels(r, g, b)
    return im


class Filter(object):
    IDENTITY = 'identity'
    GOTHAM = 'gotham'
    CLARENDON = 'clarendon'
    GINGHAM = 'gingham'
    JUNO = 'juno'
    LARK = 'lark'
    REYES = 'reyes'

    _FILTER_MAP = {
        IDENTITY: lambda im: im,
        GOTHAM: _gotham,
        CLARENDON: _clarendon,
        GINGHAM: _gingham,
        JUNO: _juno,
        LARK: _lark,
        REYES: _reyes
    }

    FILTER_TYPES = [IDENTITY, GOTHAM, CLARENDON, GINGHAM, JUNO, LARK, REYES]

    def __init__(self, filter_type=IDENTITY):
        assert filter_type in self._FILTER_MAP
        self.filter_type = filter_type

    def apply(self, image):
        return self._FILTER_MAP[self.filter_type](image)

    def to_categorical(self):
        return keras.utils.to_categorical(self.FILTER_TYPES.index(self.filter_type), len(self.FILTER_TYPES))

    @classmethod
    def from_categorical(cls, categorical, threshold=0.7):
        max_conf_idx = np.argmax(categorical)
        if categorical[max_conf_idx] >= 0.7:
            return Filter(cls.FILTER_TYPES[max_conf_idx])
        else:
            return None
