import os
from dataclasses import dataclass
import glob
from functools import cached_property

import numpy as np
import tensorflow as tf
import pandas as pd

# path issues with sphinx and the relative paths for running as a module, as intended when I wrote them
abs_path = os.path.dirname(__file__)


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generator for Keras training to allow multiprocessing and training on batches with only the
    batch itself being loaded into memory.

    This implementation was heavily inspired by various examples for creating data generators
    for keras, namely https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
    """

    def __init__(
        self,
        data_filenames: np.array,
        labels: pd.Series,
        n_classes: int,
        batch_size: int = 64,
        dim: tuple = (32, 32, 5),
    ):
        self.data_filenames = data_filenames
        self.labels = labels
        self.n_classes = n_classes
        self.dim = dim
        self.batch_size = batch_size
        self.batch = self.get_batch()
        self.indexes = self.get_indices()
        self.shuffle = True

    def get_batch(self):
        """Generating upon initalization, so we can keep things thread-safe for multiprocessing"""
        shuffled = np.random.choice(self.data_filenames, len(self.data_filenames), replace=False)
        batch = {}
        num_batches = len(self.data_filenames) // self.batch_size
        j = 1
        for i in range(0, num_batches):
            batch[i] = shuffled[i * self.batch_size : j * self.batch_size]
            j += 1

        return batch

    def get_indices(self):
        """Returns indices for __getitem__"""
        return np.arange(len(self.data_filenames) // self.batch_size)

    def __len__(self):
        """Number of batches per epoch"""
        return len(self.data_filenames) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data from batch files
        X, y = self._data_generation(self.batch[index])

        return X, y

    def on_epoch_end(self):
        """Updates indexes with shuffle after each epoch"""
        print(
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\
               Calling on_epoch_end in DataGenerator \n\
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        )
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _data_generation(self, batch):
        """
        From file names for a batch, generate the data.
        """
        # print("Is this getting called??")
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size))

        # Generate data
        for i, ID in enumerate(batch):
            x = glob.glob("data/processed/" + str(ID) + "-*.npy")[0]
            # print(x)
            X[i] = np.load(os.path.join(abs_path, "..", x)).T

            # small bug here from how I've integrated the model architecture from Pasquet.
            if type(self.labels[ID]) == pd.Series:
                y[i] = self.labels[ID].values[0]
            else:
                y[i] = self.labels[ID]

        return X, y
