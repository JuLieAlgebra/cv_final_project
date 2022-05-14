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

from final_project import preprocessing
from final_project import salted
from final_project.data_generator import DataGenerator
from final_project.models import InceptionModel


###################
abs_path = os.path.dirname(__file__)  # for pathing with sphinx
join = os.path.join  # for my own sanity
###################


class TrainModel(luigi.Task):
    """
    Luigi task to train model with generator
    Reads in config files to determine parameters of next experiment run and creates
    Model and generator objects for training and validation.

    :param batch_size: int - specified via config file
    :param test_split: float - specified via config file
    :param num_classes: int - specified via config file
    :param lr: float - specified via config file
    :param seed: int - specified via config file
    """

    # Two config files to avoid code changes
    conf_name = OmegaConf.load(join(abs_path, "conf", "train_configs.yaml"))["conf_to_use"]
    params = OmegaConf.load(join(abs_path, "conf", conf_name))

    # some repeated code in order to have the parameters become part of the salt
    batch_size = luigi.Parameter(default=params["batch_size"])
    test_split = luigi.Parameter(default=params["test_split"])  # fraction of data to be in test set
    num_classes = luigi.Parameter(default=params["num_classes"])
    lr = luigi.Parameter(default=params["lr"])
    seed = luigi.Parameter(default=params["random_seed"])
    # input image shape isn't loaded here because the modifiable thing, the individual image shape, is
    # already affecting the salt for PreProcessing

    def output(self) -> luigi.LocalTarget:
        """The 'I'm done' success file will be the trained model"""
        return luigi.LocalTarget(join(abs_path, "..", "data", "models", f"_SUCCESS-{salted.get_salted_version(self)}"))

    def run(self) -> None:
        """Handles kicking off model training and writing success files/final model file"""
        self.model = InceptionModel(**self.params)
        history = self.model.train(
            checkpoint_path=join(abs_path, "..", "data", "models", f"checkpoint-{salted.get_salted_version(self)}")
        )

        self.model.model.save(join(abs_path, "..", "data", "models", f"model-{salted.get_salted_version(self)}"))

        with self.output().open(mode="w") as success:
            success.write("success")
