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
    """

    def __init__(
        self,
        data_IDs: np.array,
        labels: pd.Series,
        n_classes: int,
        batch_size: int = 64,
        dim: tuple = (32, 32, 5),
    ):
        self.data_IDs = data_IDs
        self.labels = labels
        self.n_classes = n_classes
        self.dim = dim
        self.batch_size = batch_size
        self.batch = self.get_batch()
        self.indexes = self.get_indices()
        self.shuffle = True

    def get_batch(self):
        """Generating upon initalization, so we can keep things thread-safe for multiprocessing"""
        batch = {}
        num_batches = len(self.data_IDs) // self.batch_size
        print("Leftover: ", len(self.data_IDs) % self.batch_size)
        j = 1
        for i in range(num_batches):
            batch[i] = self.data_IDs[i * self.batch_size : j * self.batch_size]
            j += 1

        return batch

    def get_indices(self):
        """Returns indices for __getitem__"""
        return np.arange(len(self.data_IDs) // self.batch_size)

    def __len__(self):
        """Number of batches per epoch"""
        return len(self.data_IDs) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate data from batch files
        X, y = self._data_generation(self.batch[index])

        return X, y

    def on_epoch_end(self):
        """Updates indexes with shuffle after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, batch):
        """
        From file names for a batch, generate the data.
        """
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size))

        # Generate data
        for i, ID in enumerate(batch):
            x = glob.glob("data/processed/" + str(ID) + "-*.npy")[0]
            # print(x)
            X[i] = np.transpose(np.load(os.path.join(abs_path, "..", x)), (1, 2, 0))

            # ID is not entirely unique in the csv, so occasionally I get one objID with two spectroscopic redshifts
            # I decided to handle this by averaging the estimates
            if type(self.labels[ID]) == pd.Series:
                y[i] = np.mean(self.labels[ID].values)
            else:
                y[i] = self.labels[ID]

        return X, y
