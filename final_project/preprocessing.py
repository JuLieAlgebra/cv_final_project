import os
import zipfile

import luigi
import numpy as np
import omegaconf
import pandas as pd

from final_project import data_downloader
from final_project import preprocessing_utils
from final_project import salted


class Preprocessing(luigi.Task):
    """Takes the zipped image files and ends up with the finished, saved datacubes in .npy format"""

    __version__ = "0.1.0"

    # Grabs the tabular data's name, want to be agnostic to future changes to that file
    tabular_path = omegaconf.OmegaConf.load("final_project/conf/aws_paths.yaml")["tabular_data"]
    local_paths = omegaconf.OmegaConf.load("final_project/conf/local_paths.yaml")
    processed_path = os.path.join(local_paths["data"], local_paths["processed"])

    # range of downloaded files to process = [lower, upper]
    lower = luigi.IntParameter()
    upper = luigi.IntParameter()

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            os.path.join(self.processed_path, f"_SUCCESS{self.lower}-{self.upper}-{salted.get_salted_version(self)}")
        )

    def requires(self) -> list:
        return data_downloader.ImageDownloader(lower=self.lower * 5, upper=self.upper * 5)

    def run(self):
        """Big workhorse."""
        df = pd.read_csv(os.path.join(self.local_paths["data"], self.tabular_path))
        rows = df.iloc[self.lower : self.upper]

        for i, observation in rows.iterrows():
            data_cube = preprocessing_utils.get_data_cube(observation)
            file_name = os.path.join(self.processed_path, f"{observation.id}-{salted.get_salted_version(self)}.npy")

            # can I improve this?
            data_cube.save(file_name)

        # writing the success file
        with self.output().open("w") as outfile:
            outfile.write(f"Success: {self.lower}-{self.upper}")


class GenTrainTestData(luigi.Task):
    """Skeleton code for possibly how I'll do the test & train split and prep before training"""

    local_paths = omegaconf.OmegaConf.load("final_project/conf/local_paths.yaml")

    test_directory = os.path.join(local_paths["data"], local_paths["test"])
    train_directory = os.path.join(local_paths["data"], local_paths["train"])
    SPLIT = 0.8  # fraction of dataset that is trained on

    def output(self):
        # Oh not going to work, outputs are entire directories? Need to resolve. Also with the Salted stuff...
        paths = ...
        return {"test": luigi.LocalTarget(self.test_directory), "train": luigi.LocalTarget(self.train_directory)}

    def requires(self):
        """ """
        return final_project.preprocessing.ToNumpy()
