"""
Filter inversion models.

Author: Cem Gokmen
"""


import numpy as np
import keras
import utility
import os
from data.filters import Filter
import multiprocessing
import functools


DEFAULT_MODEL_FOLDER = os.path.join(utility.ROOT_DIR, 'data/inversion_models')


class Inverter(object):
    def __init__(self, model_folder=DEFAULT_MODEL_FOLDER):
        self.model_folder = model_folder
        self.models = {}
        self._load_models()

    def _model_path(self, filter_type):
        return os.path.join(self.model_folder, "{}_model.h5".format(filter_type))

    def _dataset_path(self, filter_type):
        return os.path.join(self.model_folder, "{}_dataset.npz".format(filter_type))

    def _load_models(self):
        print("Loading inversion models... Might take a while")

        # Don't generate model for identity op.
        all_filter_types = Filter.FILTER_TYPES.copy()
        del all_filter_types[all_filter_types.index(Filter.IDENTITY)]

        def load_model(filter_type):
            model_path = self._model_path(filter_type)
            if os.path.exists(model_path):
                return keras.models.load_model(model_path), filter_type

            return None, filter_type

        # Try to load models.
        trained_models = [load_model(filter_type) for filter_type in all_filter_types]
        for model, filter_type in trained_models:
            self.models[filter_type] = model

        # Generate datasets for missing models.
        process_pool = multiprocessing.Pool(processes=8)

        missing_filters = [Filter(filter_type) for model, filter_type in trained_models if model is None]
        missing_dataset_names = [self._dataset_path(filter.filter_type) for filter in missing_filters]
        missing_datasets = process_pool.starmap(
            _dataset,
            zip(missing_dataset_names, missing_filters)
        )

        # Generate missing models.
        for filter, dataset in zip(missing_filters, missing_datasets):
            print("Training inversion model for {}".format(filter.filter_type))

            X, y = dataset
            history, model = _train_model(_create_model(), X, y)
            model.save(self._model_path(filter.filter_type))

            self.models[filter.filter_type] = model

        process_pool.terminate()
        print("Loaded all models.")

    def invert(self, image, filter):
        if filter.filter_type == Filter.IDENTITY:
            return image

        model = self.models[filter.filter_type]

        print("Inverting {} image with filter {}".format(image.shape, filter.filter_type))

        # We split the filtered image into 3x3 windows and store all of those.
        # We disregard the edge pixels, unfortunately.
        data_points = np.zeros((image.shape[0] - 2, image.shape[1] - 2, 3, 3, 3))

        for cent_y in range(0, data_points.shape[0]):
            for cent_x in range(0, data_points.shape[1]):
                # Get the window
                orig_y = cent_y + 1
                orig_x = cent_x + 1

                sample = image[orig_y - 1:orig_y + 2, orig_x - 1:orig_x + 2, :]
                data_points[cent_y, cent_x] = sample

        # We run the predictions
        outData = np.clip(model.predict(data_points.reshape(-1, 3, 3, 3)), 0, 1)

        # And we convert the predictions to the correct shape
        unfilteredIm = outData.reshape(data_points.shape[0], data_points.shape[1], 3)

        # Add back the edge rows/columns
        outIm = np.copy(image)
        np.copyto(outIm[1:-1, 1:-1, :], unfilteredIm)

        print("Done inverting {} image with filter {}".format(image.shape, filter.filter_type))

        return outIm


def _dataset(filename, filter, img_count=1000000):
    """
    Loads or generates a random set of 3x3 patches for training.
    """
    try:
        # Attempt to load the dataset.
        with np.load(filename) as data:
            X = data['arr_0']
            y = data['arr_1']
    except:
        # The dataset does not exist, so we regenerate.

        # Set up a sample of random images:
        sample_size = (img_count, 3, 3, 3)  # 3x3 windows, each containing 3 channels
        images = np.random.random(sample_size)

        # The correct label for each "image" is the color at its center
        y = images[:, 1, 1, :]

        # Now we apply the filter to each of our images and store the filtered image
        print("Generating dataset:")

        X = np.zeros(images.shape)

        for i in range(images.shape[0]):
            thisImg = images[i]
            filtered = filter.apply(thisImg)
            X[i] = filtered

            if (i + 1) % (img_count / 100) == 0:
                print("%s: %d%% done" % (filename, 100 * (i + 1) / img_count))

        print("Dataset generation complete.")

        np.savez(filename, X, y)

    return X[:img_count], y[:img_count]


def _create_model(lr=0.001):
    """
    Creates an untrained model.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(3, 3, 3)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3)
    ])

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=lr), metrics=['mae'])
    model.summary()

    return model


def _train_model(model, X, y, epochs=20):
    """
    Trains a model.

    (See `create_model` and `dataset`.)
    """

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('.', end='')

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    # Now actually train the model
    return (model.fit(
        X,
        y,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop, PrintDot()]
    ), model)
