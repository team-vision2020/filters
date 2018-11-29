"""
End-to-end detection & inversion.
Author: Mert Dumenci
"""

import os
import keras.models
import utility
from data.filters import Filter
from inversion.invert import Inverter
from scipy.stats import mode
import detection.histogram
import numpy as np
import random
import matplotlib.pyplot as plt


"""Filter classifier trained on MiniPlaces."""
NNFilterClassifier = keras.models.load_model(os.path.join(utility.ROOT_DIR, "detection/FullNNFilter3.h5"))

"""Filter inverter"""
inverter = Inverter()


def split_img(X):
    Xsmall = []
    for j in range(0, X.shape[1], X.shape[1] // 4):
        for k in range(0, X.shape[1], X.shape[1] // 4):
            block = X[j : j + 32, k : k + 32]
            Xsmall.append(block)

    Xsmall = np.array(Xsmall)
    return Xsmall


def prediction_voting(predict):
    mapping = ["identity", "clarendon", "gingham", "juno", "lark", "gotham", "reyes"]
    votes = np.argmax(predict, axis=1)
    vote = mode(votes).mode[0]

    return mapping[vote]


def _detect(images, threshold):
    # Threshold is unused in the new model, as it returns the best vote.
    filters = []

    for image in images:
        splits = split_img(image * 255.0)
        pred_vectors = NNFilterClassifier.predict(splits)
        best_guess = prediction_voting(pred_vectors)

        filters.append(Filter(best_guess))

    return filters


def invert(images, filters, threshold):
    def invert_info(i, image, filter):
        if i % 20 == 0:
            print("Inverting image {}/{}".format(i + 1, len(images)))
        return inverter.invert(image, filter)
    if threshold is not None:
        filters_d = _detect(images, threshold)
        
        same = 0
        for f, f_p in zip(filters, filters_d):
            same += (f.filter_type == f_p.filter_type)

        detection_acc = same / len(filters)
        print("Detected with accuracy: {}".format(detection_acc))

        filters = filters_d

    return [invert_info(i, image, filter) for i, image, filter in zip(range(len(images)), images, filters)], filters


def roundtrip_sad(ims, f_ims, filters, threshold):
    print("Computing roundtrip SAD")
    i_ims, p_filters = invert(f_ims, filters, threshold)

    total_sad = 0
    max_sad, max_sad_i = 0, 0
    min_sad, min_sad_i = 0, 0

    for i, im, filter, i_im in zip(range(len(ims)), ims, p_filters, i_ims):
        plt.imsave(os.path.join(utility.ROOT_DIR, 'output/{}_{}_inverted.png'.format(i, filter.filter_type)), i_im)
        sad = np.sum(np.absolute(im - i_im))
        total_sad += sad

        if sad > max_sad:
            max_sad = sad
            max_sad_i = i

        if sad < min_sad:
            min_sad = sad
            min_sad_i = i

    mean_sad = total_sad / len(ims)
    pp_mean_sad = mean_sad / ims[0].size

    print("Roundtrip per-pixel SAD error for {} images: {}".format(len(ims), pp_mean_sad))
    print("Max SAD: {},{}, min SAD: {}, {}".format(max_sad, max_sad_i, min_sad, min_sad_i))

    return pp_mean_sad


def baseline_sad(ims, f_ims, filters):
    total_sad = 0
    for i, f_im, im, filter in zip(range(len(f_ims)), f_ims, ims, filters):
        plt.imsave(os.path.join(utility.ROOT_DIR, 'output/{}_{}.png'.format(i, filter.filter_type)), f_im)
        plt.imsave(os.path.join(utility.ROOT_DIR, 'output/{}.png'.format(i)), im)
        total_sad += np.sum(np.absolute(f_im - im))

    mean_sad = total_sad / len(ims)
    pp_mean_sad = mean_sad / ims[0].size

    print("Baseline per-pixel SAD error for {} images: {}".format(len(ims), pp_mean_sad))

    return pp_mean_sad


SAMPLES = 900
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

    rt = roundtrip_sad(ims, f_ims, filters, threshold=threshold)
    bs = baseline_sad(ims, f_ims, filters)

    return rt, bs


# 0.7 is the best threshold [0.7, 0.95] (Experimentally found with `search_for_threshold` on that range)
run_with_threshold(0.7)
# Run with known filters, bypassing the detection step.
# run_with_threshold(None)
