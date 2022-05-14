import os
from dataclasses import dataclass
from functools import cached_property
import glob

import luigi
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# tensorflow imports
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, concatenate, Dense, PReLU, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from final_project.data_generator import DataGenerator

###################
abs_path = os.path.dirname(__file__)  # for pathing with sphinx
join = os.path.join  # for my own sanity
###################


@dataclass
class BaseModel:
    """
    Base class for Model for this pipeline. Uses lazy-loading for most heavy-lifting attributes

    :param batch_size: int - specified via config file
    :param test_split: float - specified via config file
    :param num_classes: int - specified via config file
    :param lr: float - specified via config file
    :param seed: int - specified via config file
    :param input_img_shape: tuple - specified via config file
    """

    batch_size: int
    test_split: float
    num_classes: int
    seed: int
    lr: float
    input_img_shape: tuple

    @cached_property
    def labels(self) -> dict:
        """
        Creates dict of series object, with objID as the index for easy partition access to
        labels.
        :return: dict partitioned by 'test' and 'train' of pandas Series with labels
        """
        tabular_name = OmegaConf.load(join(abs_path, "conf", "aws_paths.yaml"))["tabular_data"]
        tabular_path = join(abs_path, "..", "data", tabular_name)
        df = pd.read_csv(tabular_path, usecols=["objID", "z"])

        series = pd.Series(data=df.z.values, index=df.objID, name="z")

        return {"test": series.loc[self.partition["test"]], "train": series.loc[self.partition["train"]]}

    @cached_property
    def partition(self) -> dict:
        """
        Creates the partition labels for testing & training
        :return: dict of string file names partitioned by 'test' and 'train'
        """

        partition = {}
        # get a numpy array of all the processed data ready for training
        dataset = glob.glob(join("data", "processed", "*.npy"))
        func = lambda file: int(file[15:-15])
        dataset = np.array(list(map(func, dataset)))

        unique_data = np.unique(dataset)

        # random shuffle based on set seed.
        np.random.shuffle(unique_data)  # np.random.choice(unique_data, len(unique_data), replace=False)
        partition["test"] = unique_data[: int(len(unique_data) * self.test_split)]
        partition["train"] = unique_data[int(len(unique_data) * self.test_split) :]

        return partition

    @cached_property
    def training_generator(self) -> DataGenerator:
        """
        Cached property, so only gets called once when the property is accessed. Part of BaseModel class.
        Creates an intance of the DataGenerator class with the relevant test/train partition to use for training
        or prediction.
        """
        return DataGenerator(self.partition["train"], self.labels["train"], self.num_classes)  # , **params)

    @cached_property
    def testing_generator(self) -> DataGenerator:
        """
        Cached property, so only gets called once when the property is accessed. Part of BaseModel class.
        Creates an intance of the DataGenerator class with the relevant test/train partition to use for training
        or prediction.
        """
        return DataGenerator(self.partition["test"], self.labels["test"], self.num_classes)  # , **params)

    def train(self, checkpoint_path, epochs=20, workers=6):
        """
        Wrapper for fitting the model with the generator we created
        Note we can only use multiprocessing because we're passing in a generator

        :param epochs: int
        :param workers: int
        """
        my_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=5, verbose=1, mode="auto"),
        ]

        history = self.model.fit(
            self.training_generator,
            validation_data=self.testing_generator,
            epochs=epochs,
            use_multiprocessing=True,
            callbacks=my_callbacks,
        )
        return history

    def loss(self, true, pred):
        """
        Custom loss function for keras
        True should be a float, pred should be a (num_classes,) shaped array.
        Calculates the MSE between the true value and the expected value of the output
        """
        return tf.square(tf.subtract(true, tf.reduce_sum(tf.multiply(self.bin_values, pred))))

    @cached_property
    def bin_values(self):
        """returns array with the same length as the number of classes, with the midpoint of the start and end range
        stored (ie, the bin containing redshift values 0 to 0.2 will have bin value 0.1)
        """
        max_val = 0.4  # since 0.39.. is the max redshift value that we see
        one_step = (max_val) / self.num_classes
        bins = np.linspace(0, max_val, self.num_classes, endpoint=False, dtype=np.float32)
        bin_values = bins + one_step * 1 / 2
        return bin_values


@dataclass
class InceptionModel(BaseModel):
    """
    Inception model from Pasquet et al paper. Wrapper for many tensorflow/keras functions.

    :param batch_size: int - specified via config file
    :param test_split: float - specified via config file
    :param num_classes: int - specified via config file
    :param lr: float - specified via config file
    :param seed: int - specified via config file
    :param input_img_shape: tuple - specified via config file
    """

    batch_size: int
    test_split: float
    num_classes: int
    seed: int
    lr: float
    input_img_shape: tuple

    @cached_property
    def model(self):
        """
        Constructs the model architecture. The keras code here is all from the Inception model
        in https://github.com/umesh-timalsina/redshift/blob/3c11608d5818ae58ab2ce084832b56766858b3a1/model/model.py.
        Which is the code repository for the Pasquet et al 2018 paper
        """
        # Input Layer Galactic Images
        image_input = Input(shape=self.input_img_shape[1:])
        # Convolution Layer 1
        conv_1 = Conv2D(64, kernel_size=(5, 5), padding="same", activation=PReLU())
        conv_1_out = conv_1(image_input)

        # Pooling Layer 1
        pooling_layer1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same")
        pooling_layer1_out = pooling_layer1(conv_1_out)

        # Inception Layer 1
        inception_layer1_out = self.add_inception_layer(pooling_layer1_out, num_f1=48, num_f2=64)

        # Inception Layer 2
        inception_layer2_out = self.add_inception_layer(inception_layer1_out, num_f1=64, num_f2=92)

        # Pooling Layer 2
        pooling_layer2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same")
        pooling_layer2_out = pooling_layer2(inception_layer2_out)

        # Inception Layer 3
        inception_layer3_out = self.add_inception_layer(pooling_layer2_out, 92, 128)

        # # # Inception Layer 4
        # inception_layer4_out = self.add_inception_layer(inception_layer3_out, 92, 128)

        # Pooling Layer 3
        pooling_layer3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same")
        pooling_layer3_out = pooling_layer3(inception_layer3_out)  # modified 4 -> 3

        # Inception Layer 5
        inception_layer5_out = self.add_inception_layer(pooling_layer3_out, 92, 128, kernel_5=False)

        # input_to_pooling = cur_inception_in
        input_to_dense = Flatten(data_format="channels_last")(inception_layer5_out)
        model_output = Dense(units=self.num_classes, activation="softmax")(
            Dense(units=self.num_classes, activation="relu")(input_to_dense)
        )

        ### Construction, summary, and compiling
        model = tensorflow.keras.Model(inputs=[image_input], outputs=model_output)

        model.summary()
        # print(model.summary())
        opt = Adam(self.lr)
        model.compile(optimizer=opt, loss=self.loss)  # , metrics=["loss"])
        # from tensorflow.keras.utils import plot_model
        # plot_model(model, to_file='model.png')
        return model

    def add_inception_layer(self, input_weights, num_f1, num_f2, kernel_5=True):
        """
        These convolutional layers take care of the inception layer.
        This function is also not my code, also directly from the Pasquet et al paper repo.
        """
        # Conv Layer 1 and Layer 2: Feed them to convolution layers 5 and 6
        c1 = Conv2D(num_f1, kernel_size=(1, 1), padding="same", activation=PReLU())
        c1_out = c1(input_weights)
        if kernel_5:
            c2 = Conv2D(num_f1, kernel_size=(1, 1), padding="same", activation=PReLU())
            c2_out = c2(input_weights)

        # Conv Layer 3 : Feed to pooling layer 1
        c3 = Conv2D(num_f1, kernel_size=(1, 1), padding="same", activation=PReLU())
        c3_out = c3(input_weights)

        # Conv Layer 4: Feed directly to concat
        c4 = Conv2D(num_f2, kernel_size=(1, 1), padding="same", activation=PReLU())
        c4_out = c4(input_weights)

        # Conv Layer 5: Feed from c1, feed to concat
        c5 = Conv2D(num_f2, kernel_size=(3, 3), padding="same", activation=PReLU())
        c5_out = c5(c1_out)

        # Conv Layer 6: Feed from c2, feed to concat
        if kernel_5:
            c6 = Conv2D(num_f2, kernel_size=(5, 5), padding="same", activation=PReLU())
            c6_out = c6(c2_out)

        # Pooling Layer 1: Feed from conv3, feed to concat
        p1 = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")
        p1_out = p1(c3_out)

        if kernel_5:
            return concatenate([c4_out, c5_out, c6_out, p1_out])
        else:
            return concatenate([c4_out, c5_out, p1_out])


@dataclass
class MixedInputModel(BaseModel):
    """
    Stump for next model class that I'll implement, from the new state of the art paper
    Henghes et al, 2022 (just recently accepted to Astronomy & Astrophysics, I think).

    :param batch_size: int - specified via config file
    :param test_split: float - specified via config file
    :param num_classes: int - specified via config file
    :param lr: float - specified via config file
    :param seed: int - specified via config file
    :param input_img_shape: tuple - specified via config file
    """

    batch_size: int
    test_split: float
    num_classes: int
    seed: int
    lr: float
    input_img_shape: tuple

    @cached_property
    def model(self):
        """Constructs the model architecture"""
        pass
