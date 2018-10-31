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
import matplotlib.pyplot as plt


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


def roundtrip_sad(ims, f_ims):
    print("Computing roundtrip SAD")
    i_ims = invert(f_ims)

    total_sad = 0
    for i, im, i_im in zip(range(len(ims)), ims, i_ims):
        # plt.imsave("/Users/mert/Downloads/inverted_{}.png".format(i), i_im)
        total_sad += np.sum(np.absolute(im - i_im))

    mean_sad = total_sad / len(ims)
    pp_mean_sad = mean_sad / ims[0].size

    print("Roundtrip per-pixel SAD error for {} images: {}".format(len(ims), pp_mean_sad))


def baseline_sad(ims, f_ims):
    print("Computing baseline SSD")

    total_sad = 0
    for i, f_im, im in zip(range(len(f_ims)), f_ims, ims):
        # plt.imsave("/Users/mert/Downloads/filtered_{}.png".format(i), f_im)
        # plt.imsave("/Users/mert/Downloads/baseline_{}.png".format(i), im)
        total_sad += np.sum(np.absolute(f_im - im))

    mean_sad = total_sad / len(ims)
    pp_mean_sad = mean_sad / ims[0].size

    print("Baseline per-pixel SAD error for {} images: {}".format(len(ims), pp_mean_sad))


SAMPLES = 100
APPLICABLE_FILTERS = Filter.FILTER_TYPES[1:]  # Ignore the `identity` filter.


def random_filter():
    type = random.choice(APPLICABLE_FILTERS)
    return Filter(type)


filters = [random_filter() for _ in range(SAMPLES)]
miniplaces_root = os.path.join(utility.ROOT_DIR, 'data/miniplaces/val')
miniplaces_paths = random.sample(list(os.listdir(miniplaces_root)), SAMPLES)

print("Loading {} ims".format(SAMPLES))
ims = [utility.image_from_path(os.path.join(utility.ROOT_DIR, 'data/miniplaces/val', path))
       for path in miniplaces_paths]
print("Filtering {} ims".format(SAMPLES))
f_ims = [filter.apply(im) for filter, im in zip(filters, ims)]

roundtrip_sad(ims, f_ims)
baseline_sad(ims, f_ims)