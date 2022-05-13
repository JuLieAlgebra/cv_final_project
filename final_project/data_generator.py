import os
from dataclasses import dataclass
import glob

import numpy as np
import tensorflow as tf

# path issues with sphinx and the relative paths for running as a module, as intended when I wrote them
abs_path = os.path.dirname(__file__)


@dataclass
class DataGenerator(tf.keras.utils.Sequence):
    """
    Generator for Keras training to allow multiprocessing and training on batches with only the
    batch itself being loaded into memory.

    This implementation was heavily inspired by various examples for creating data generators
    for keras, namely https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
    """

    data_filenames: list
    labels: list
    n_classes: int
    batch_size: int = 32
    dim: tuple = (32, 32, 5)

    def __len__(self):
        """Number of batches per epoch"""
        return len(self.data_filenames) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        # For batch number index, get the file names
        batch = np.random.choice(self.data_filenames, self.batch_size, replace=False)

        # Generate data from batch files
        X, y = self._data_generation(batch)

        return X, y

    def on_epoch_end(self):
        """Updates indexes with shuffle after each epoch"""
        self.indexes = np.arange(len(self.data_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        # check accuracy, if better, save?

    def _data_generation(self, batch):
        """From file names for a batch, generate the data"""
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(batch):
            x = glob.glob("data/processed/" + str(ID) + "-*.npy")[0]
            X[
                i,
            ] = np.load(x).T

            # small bug here from how I've integrated the model architecture from Pasquet.
            if type(self.labels[ID]) != float and type(self.labels[ID]) != np.float64:
                print(type(self.labels[ID]))
            else:
                y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y * self.n_classes, num_classes=self.n_classes)
        # np.array(list(map(np.argwhere, tf.keras.utils.to_categorical(y*self.n_classes, num_classes=self.n_classes))))


# From
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
