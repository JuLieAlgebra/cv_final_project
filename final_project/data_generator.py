import os
from dataclasses import dataclass
import glob

import numpy as np
import tensorflow as tf

# path issues with sphinx and the relative paths for running as a module, as intended when I wrote them
abs_path = os.path.dirname(__file__)


@dataclass
class DataGenerator(tf.keras.utils.Sequence):  # utils.Sequence):
    """Generator for Keras training with model.fit_generator"""

    data_filenames: list
    labels: list
    n_classes: int
    batch_size: int = 32
    dim: tuple = (5, 32, 32)

    def __len__(self):
        """Denotes the number of batches per epoch"""
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
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(batch):
            try:  # TODO remove, artifact of not every obj in csv being downloaded & processed
                # Store sample
                x = glob.glob("data/processed/" + str(ID) + "-*.npy")[0]
                X[
                    i,
                ] = np.load(x)

                # Store class
                y[i] = self.labels[ID]
            except KeyError:
                pass
        print("KERA S UTTILS CATEGORIACAL : ", tf.keras.utils.to_categorical(y, num_classes=self.n_classes))
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


# import numpy as np

# from tf.keras.models import Sequential
# from my_classes import DataGenerator

# # Parameters
# params = {'dim': (32,32,32),
#           'batch_size': 64,
#           'n_classes': 6,
#           'n_channels': 1,
#           'shuffle': True}

# # Datasets
# partition = # IDs
# labels = # Labels

# # Generators
# training_generator = DataGenerator(partition['train'], labels, **params)
# validation_generator = DataGenerator(partition['validation'], labels, **params)

# # Design model
# model = Sequential()
# [...] # Architecture
# model.compile()

# # Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)


# From
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""
class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(self, list_IDs, labels, batch_size=32, dim=(5, 32, 32), n_channels=1, n_classes=100, shuffle=True):
        "Initialization"
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self._data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[
                i,
            ] = np.load("data/" + ID + ".npy")

            # Store class
            y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
"""
