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


def _detect(images, threshold):
    hists = detection.histogram.three_channel_histogram(images)
    pred_vectors = NNFilterClassifier.predict(hists)
    filters = [Filter.from_categorical(prediction, threshold) for prediction in pred_vectors]

    return filters


def invert(images, filters, threshold):
    def invert_info(i, image, filter):
        if i % 20 == 0:
            print("Inverting image {}/{}".format(i + 1, len(images)))
        return inverter.invert(image, filter)
    if filters is None:
        filters = _detect(images, threshold)
    return [invert_info(i, image, filter) for i, image, filter in zip(range(len(images)), images, filters)]


def roundtrip_sad(ims, f_ims, filters, threshold):
    print("Computing roundtrip SAD")
    i_ims = invert(f_ims, filters, threshold)

    total_sad = 0
    for i, im, i_im in zip(range(len(ims)), ims, i_ims):
        # plt.imsave("/Users/mert/Downloads/inverted_{}.png".format(i), i_im)
        total_sad += np.sum(np.absolute(im - i_im))

    mean_sad = total_sad / len(ims)
    pp_mean_sad = mean_sad / ims[0].size

    print("Roundtrip per-pixel SAD error for {} images: {}".format(len(ims), pp_mean_sad))

    return pp_mean_sad


def baseline_sad(ims, f_ims):
    total_sad = 0
    for i, f_im, im in zip(range(len(f_ims)), f_ims, ims):
        # plt.imsave("/Users/mert/Downloads/filtered_{}.png".format(i), f_im)
        # plt.imsave("/Users/mert/Downloads/baseline_{}.png".format(i), im)
        total_sad += np.sum(np.absolute(f_im - im))

    mean_sad = total_sad / len(ims)
    pp_mean_sad = mean_sad / ims[0].size

    print("Baseline per-pixel SAD error for {} images: {}".format(len(ims), pp_mean_sad))

    return pp_mean_sad


SAMPLES = 10000
APPLICABLE_FILTERS = Filter.FILTER_TYPES[1:]  # Ignore the `identity` filter.


def random_filter():
    type = random.choice(APPLICABLE_FILTERS)
    return Filter(type)


miniplaces_root = os.path.join(utility.ROOT_DIR, 'data/miniplaces/val')
miniplaces_paths = random.sample(list(os.listdir(miniplaces_root)), SAMPLES)
print("Loading {} ims".format(SAMPLES))
ims = [utility.image_from_path(os.path.join(utility.ROOT_DIR, 'data/miniplaces/val', path))
       for path in miniplaces_paths]


def search_for_threshold():
    # Search for correct threshold.
    best_ratio = 0
    best_threshold = 0

    for t in np.linspace(0.7, 0.95, 20):
        total_ratio = 0
        ITERATIONS = 3

        for i in range(ITERATIONS):
            print("Iteration {}/{} for threshold {}".format(i + 1, ITERATIONS, t))
            rt, bs = run_with_threshold(t)
            total_ratio += bs / rt

        mean_ratio = total_ratio / ITERATIONS

        print("Mean ratio for {}: {}".format(t, mean_ratio))
        if mean_ratio > best_ratio:
            best_ratio = mean_ratio
            best_threshold = t

    print("Best threshold: {}".format(best_threshold))


def run_with_threshold(threshold):
    filters = [random_filter() for _ in range(SAMPLES)]
    print("Filtering {} ims with randomly selected filters".format(SAMPLES))
    f_ims = [filter.apply(im) for filter, im in zip(filters, ims)]

    rt = roundtrip_sad(ims, f_ims, None if threshold else filters, threshold=threshold)
    bs = baseline_sad(ims, f_ims)

    return rt, bs


# 0.7 is the best threshold [0.7, 0.95] (Experimentally found with `search_for_threshold` on that range)
run_with_threshold(0.7)
# Run with known filters, bypassing the detection step.
run_with_threshold(None)
