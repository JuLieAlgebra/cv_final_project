import os
from dataclasses import dataclass

import luigi
from tensorflow import keras
import omegaconf

from final_project import preprocessing
from final_project import salted
from final_project.data_generator import DataGenerator

# path issues with sphinx and the relative paths for running as a module, as intended when I wrote them
abs_path = os.path.dirname(__file__)
join = os.path.join


class TrainModel(luigi.Task):
    """
    Luigi task to train model with generator
    Reads in config files to determine parameters of next experiment run and creates
    Model and generator objects for training and validation.
    """

    # Two config files to avoid code changes
    conf_name = omegaconf.OmegaConf.load(join(abs_path, "conf", "train_configs.yaml"))["conf_to_use"]
    params = omegaconf.OmegaConf.load(join(abs_path, "conf", conf_name))

    # some repeated code in order to have the parameters become part of the salt
    batch_size = luigi.Parameter(default=params["batch_size"])
    test_split = luigi.Parameter(default=params["test_split"])  # fraction of data to be in test set
    num_classes = luigi.Parameter(default=params["num_classes"])
    seed = luigi.Parameter(default=params["random_seed"])
    np.random.seed(seed)

    # gonna use property to set this TODO
    partition = partition

    # need to make partition and correct this
    training_generator = DataGenerator(partition["train"], labels, **params)
    validation_generator = DataGenerator(partition["validation"], labels, **params)

    model = Model(**params)

    @property
    def partition(self) -> dict:
        """
        Or something like this. Creates the partition labels for testing & training
        :return: dict of file names, minus salts TODO fix
        """

        partition = {}
        # get a numpy array of all the processed data ready for training
        dataset = np.array(os.listdir("../data/processed/"))

        # TODO check the number for slicing, make sure I only grab the unique object IDs
        unique_data = np.unique(dataset[:, :-14])

        # random shuffle based on set seed.
        shuffled = np.random.shuffle(unique_data)  # np.random.choice(unique_data, len(unique_data), replace=False)
        partition["test"] = shuffled[: len(shuffled) * self.test_split]
        partition["train"] = shuffled[len(shuffled) * self.test_split :]

        return partition

    def train(self) -> None:
        """Trains the model, not sure if this is how I want it structured."""
        # Train model on dataset
        self.model.fit_generator(
            generator=self.training_generator,
            validation_data=self.validation_generator,
            use_multiprocessing=True,
            workers=6,
        )

    def predict(self) -> None:
        """Should have a function for easy predictions and easy plotting/visualization"""
        # eh?
        self.model.predict_generator(generator=self.validation_generator)

    def requires(self) -> luigi.LocalTarget:
        """Ah, need to think about how to condense back down........ Need salt to not depend on partition for everything
        in order for luigi to condense all the 10+ calls of TrainModel to a single one.... ?
        Actually no, I think I just kick off the ImageProcessing here."""
        return

    def output(self) -> luigi.LocalTarget:
        """Trained model with checkpoints....."""
        # this need to be redone, since this is pointing to a directory, not a file
        return luigi.LocalTarget(join(abs_path, "..", "data", f"models-{salted.get_salted_version(self)}"))

    def run(self) -> None:
        """Handles kicking off model training and writing success files/final model file"""
        self.model.train()

        # with self.output().open(mode='w') as success:
        #     success.write("success")


@dataclass
class InceptionModel:
    batch_size: int
    test_split: float
    num_classes: int
    seed: int

    def fit_generator(self, **kwargs):
        """Wrapper for fitting the model with the generator we created"""
        return self.model.fit_generator(kwargs)


@dataclass
class MixedInputModel:
    batch_size: int
    test_split: float
    num_classes: int
    seed: int

    def fit_generator(self, **kwargs):
        """Wrapper for fitting the model with the generator we created"""
        return self.model.fit_generator(kwargs)


# from keras.layers import Sequential
from keras import Model
from keras.layers import Input, Conv2D, AveragePooling2D, concatenate, Dense, PReLU, Flatten
from keras.optimizers import Adam

# from keras.layers.merge import concatenate
import numpy as np


class RedShiftClassificationModel(Model):
    """
    Inception model from https://github.com/umesh-timalsina/redshift/blob/3c11608d5818ae58ab2ce084832b56766858b3a1/model/model.py
    Which is the code repository for the Pasquet et al 2018 paper
    """

    def __init__(self, input_img_shape, num_redshift_classes=1024):
        """Initialize the model"""
        # Input Layer Galactic Images
        image_input = Input(shape=input_img_shape)
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

        # Inception Layer 4
        inception_layer4_out = self.add_inception_layer(inception_layer3_out, 92, 128)

        # Pooling Layer 3
        pooling_layer3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same")
        pooling_layer3_out = pooling_layer3(inception_layer4_out)

        # Inception Layer 5
        inception_layer5_out = self.add_inception_layer(pooling_layer3_out, 92, 128, kernel_5=False)

        # input_to_pooling = cur_inception_in
        input_to_dense = Flatten(data_format="channels_last")(inception_layer5_out)
        print(input_to_dense.shape)
        model_output = Dense(units=num_redshift_classes, activation="softmax")(
            Dense(units=num_redshift_classes, activation="relu")(input_to_dense)
        )

        super(RedShiftClassificationModel, self).__init__(inputs=[image_input], outputs=model_output)
        self.summary()
        opt = Adam(lr=0.001)
        self.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

    def add_inception_layer(self, input_weights, num_f1, num_f2, kernel_5=True):
        """These convolutional layers take care of the inception layer"""
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


# if __name__ == "__main__":
#     from model_utils import save_model_image
#     rscm = RedShiftClassificationModel((64, 64, 5), 1024)
#     save_model_image(rscm, 'redshiftmodel.png')
#     # print(rscm.predict(np.random.rand(1, 64, 64, 5)).shape)
#     # rscm.prepare_for_training()


# from https://machinelearningmastery.com/check-point-deep-learning-models-keras/
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # checkpoint
# filepath="weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# # Fit the model
# model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)


"""
# From https://github.com/jpasquet/Photoz/blob/master/network.py
# Need to make sure that the model architecture is the same for replication

# -*- coding: utf-8 -*-
# TensorFlow CNN model
# From "Photometric redshifts from SDSS images using a Convolutional Neural Network"
# by J.Pasquet et al. 2018

import numpy as np
import tensorflow as tf

def prelu(x):
    with tf.name_scope('PRELU'):
        _alpha = tf.get_variable("prelu", shape=x.get_shape()[-1],
        dtype = x.dtype, initializer=tf.constant_initializer(0.0))

    return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)


def conv2d(input, num_output_channels, kernel_size, name):

    with tf.variable_scope(name):
        num_in_channels = input.get_shape()[-1].value
        kernel_shape = [kernel_size,
                        kernel_size,
                        num_in_channels,
                        num_output_channels]

        biases = tf.get_variable('biases',
                                 shape=[num_output_channels],
                                 initializer=tf.constant_initializer(0.1))
        kernel =  tf.get_variable('weights',
                                  shape=kernel_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())

        outputs = tf.nn.conv2d(input,
                               kernel,
                               strides=[1,1,1,1],
                               padding="SAME")
        outputs = tf.nn.bias_add(outputs, biases)
        outputs = prelu(outputs)

    return outputs


def pool2d(input,kernel_size,stride,name):

    print(input, [1, kernel_size, kernel_size, 1],[1, stride, stride], 1)
    with tf.variable_scope(name):
        return tf.nn.avg_pool(input,
                              ksize=[1, kernel_size, kernel_size, 1],
                              strides=[1, stride, stride, 1],
                              padding="SAME",
                              name=name)


def fully_connected(input, num_outputs, name, withrelu=True):

    with tf.variable_scope(name):
        num_input_units = input.get_shape()[-1].value
        kernel_shape = [num_input_units, num_outputs]
        kernel = tf.get_variable('weights',
                                 shape=kernel_shape,
                                 initializer=tf.contrib.layers.xavier_initializer())

        outputs = tf.matmul(input, kernel)

        biases = tf.get_variable('biases',
                                 shape=[num_outputs],
                                 initializer=tf.constant_initializer(0.1))

        outputs = tf.nn.bias_add(outputs, biases)

        if withrelu:
            outputs = tf.nn.relu(outputs)

        return outputs


def inception(input, nbS1, nbS2, name, output_name, without_kernel_5=False):

    with tf.variable_scope(name):
        s1_0 = conv2d(input=input,
                      num_output_channels=nbS1,
                      kernel_size=1,
                      name=name + "S1_0")

        s2_0 = conv2d(input=s1_0,
                      num_output_channels=nbS2,
                      kernel_size=3,
                      name=name + "S2_0")

        s1_2 = conv2d(input=input,
                      num_output_channels=nbS1,
                      kernel_size=1,
                      name=name + "S1_2")

        pool0 = pool2d(input=s1_2,
                       kernel_size=2,
                       stride=1,
                       name=name + "pool0")

        if not(without_kernel_5):
            s1_1 = conv2d(input=input,
                          num_output_channels=nbS1,
                          kernel_size=1,
                          name=name + "S1_1")

            s2_1 = conv2d(input=s1_1,
                          num_output_channels=nbS2,
                          kernel_size=5,
                          name=name + "S2_1")

        s2_2 = conv2d(input=input,
                      num_output_channels=nbS2,
                      kernel_size=1,
                      name=name + "S2_2")

        if not(without_kernel_5):
            output = tf.concat(values=[s2_2, s2_1, s2_0, pool0],
                               name=output_name,
                               axis=3)
        else:
            output = tf.concat(values=[s2_2, s2_0, pool0],
                               name=output_name,
                               axis=3)

    return output


def model():

    reddening = tf.placeholder(tf.float32, shape=[None, 1], name="reddening")

    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 5], name="x")

    conv0 = conv2d(input=x, num_output_channels=64, kernel_size=5, name="conv0")
    conv0p = pool2d(input=conv0, kernel_size=2, stride=2, name="conv0p")
    i0 = inception(conv0p, 48, 64, name="I0_", output_name="INCEPTION0")
    i1 = inception(i0, 64, 92, name="I1_", output_name="INCEPTION1")
    i1p = pool2d(input=i1, kernel_size=2, name="INCEPTION1p", stride=2)
    i2 = inception(i1p, 92, 128, name="I2_", output_name="INCEPTION2")
    i3 = inception(i2, 92, 128, name="I3_", output_name="INCEPTION3")
    i3p = pool2d(input=i3, kernel_size=2, name="INCEPTION3p", stride=2)
    i4 = inception(i3p, 92,128, name="I4_", output_name="INCEPTION4",
                   without_kernel_5=True)

    flat = tf.layers.Flatten()(i4)
    concat = tf.concat(values=[flat,reddening], axis=1)

    fc0 = fully_connected(input=concat, num_outputs=1096, name="fc0")
    fc1 = fully_connected(input=fc0, num_outputs=1096, name="fc0b")
    fc2 = fully_connected(input=fc1, num_outputs=180, name="fc1",
                          withrelu=False)

    output = tf.nn.softmax(fc2)

    params = {"output": output, "x": x, "reddening": reddening}

    return params
"""
