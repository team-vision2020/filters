"""
End-to-end detection & inversion.
Author: Mert Dumenci
"""

import os
import keras.models
import utility
from data.filters import Filter
from inversion.invert import Inverter
import detection.histogram
import numpy as np
import random


"""Filter classifier trained on MiniPlaces."""
NNFilterClassifier = keras.models.load_model(os.path.join(utility.ROOT_DIR, "detection/NNFilterClassifier.h5"))

"""Filter inverter"""
inverter = Inverter()


def _detect(images):
    hists = detection.histogram.three_channel_histogram(images)
    pred_vectors = NNFilterClassifier.predict(hists)
    filters = [Filter.from_categorical(prediction) for prediction in pred_vectors]

    return filters


def invert(images):
    filters = _detect(images)
    return [inverter.invert(image, filter) for image, filter in zip(images, filters)]


def mean_ssd(samples=10000):
    def random_filter():
        type = random.choice(Filter.FILTER_TYPES)
        return Filter(type)

    miniplaces_root = os.path.join(utility.ROOT_DIR, 'data/miniplaces/val')
    miniplaces_paths = random.sample(list(os.listdir(miniplaces_root)), samples)

    ims = [utility.image_from_path(os.path.join(utility.ROOT_DIR, 'data/miniplaces/val', path))
           for path in miniplaces_paths]
    f_ims = [random_filter().apply(im) for im in ims]
    i_ims = invert(f_ims)

    total_ssd = 0
    for f_im, i_im in zip(f_ims, i_ims):
        total_ssd += np.sum((f_im - i_im) ** 2)
    mean_ssd = total_ssd / len(f_ims)
    pp_mean_ssd = mean_ssd / f_ims[0].size

    print("Mean roundtrip per-pixel SSD error for {} images: {}".format(samples, pp_mean_ssd))


mean_ssd()