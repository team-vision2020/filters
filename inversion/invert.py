from skimage import img_as_float
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop

def dataset(filename, f, img_count):
    try:
        # Attempt to load the dataset
        with np.load(filename) as data:
            X = data['arr_0']
            y = data['arr_1']
    except:
        # The dataset does not exist, so we regenerate

        # Set up a sample of random images:
        sample_size = (img_count, 3, 3, 3) # 3x3 windows, each containing 3 channels
        images = np.random.random(sample_size)

        # The correct label for each "image" is the color at its center
        y = images[:, 1, 1, :]

        # Now we apply the filter to each of our images and store the filtered image
        print("Generating dataset:")
        X = np.zeros(images.shape)
        for i in range(images.shape[0]):
            thisImg = images[i]
            filtered = f(thisImg)
            X[i] = filtered

            if ((i + 1) % (img_count / 100) == 0):
                print("%d%% done" % (100 * (i + 1) / img_count))
        print("Dataset generation complete.")

        np.savez(filename, X, y)

    return X[:img_count], y[:img_count]


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(3, 3, 3)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(3)
    ])

    optimizer = RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])

    model.summary()

    return model


def train_model(model, X, y):
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            print('.', end='')

    EPOCHS = 100

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    # Now actually train the model
    return model.fit(X, y, epochs=EPOCHS,
                        validation_split=0.1, verbose=0,
                        callbacks=[early_stop, PrintDot()])


def unfilter_image(filtIm, model):
    # We split the filtered image into 3x3 windows and store all of those.
    # We disregard the edge pixels, unfortunately
    dataPts = np.zeros((filtIm.shape[0] - 2, filtIm.shape[1] - 2, 3, 3, 3))

    for cent_y in range(0, dataPts.shape[0]):
        for cent_x in range(0, dataPts.shape[1]):
            # Get the window
            orig_y = cent_y + 1
            orig_x = cent_x + 1

            sample = filtIm[orig_y-1:orig_y+2, orig_x-1:orig_x+2, :]
            dataPts[cent_y, cent_x] = sample

    # We run the predictions
    outData = np.clip(model.predict(dataPts.reshape(-1, 3, 3, 3)), 0, 1)

    # And we convert the predictions to the correct shape
    outIm = outData.reshape(dataPts.shape[0], dataPts.shape[1], 3)

    return outIm


def save_model(filename, model):
    model.save(filename)

def load_model(filename):
    return keras.models.load_model(filename)