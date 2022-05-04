import os
import zipfile

import luigi
import numpy as np
import omegaconf

from final_project import data_downloader
from final_project import preprocessing_utils
from final_project import salted


class Preprocessing(luigi.Task):
    """Takes the zipped image files and ends up with the finish, saved datacubes in .npy format"""

    # Grabs the tabular data's name, want to be agnostic to future changes to that file
    tabular_path = omegaconf.OmegaConf.load("final_project/conf/aws_paths.yaml")["tabular_data"]

    # # Parameters for kicking off the requires tasks (which should also run in parallel)
    # n_workers = luigi.Parameter(default=10)
    # n_urls = luigi.Parameter(default=50000)

    # range of downloaded files to process = [lower, upper]
    lower = luigi.IntParameter()
    upper = luigi.IntParameter()

    def output(self) -> luigi.LocalTarget:
        """Outputs _SUCCESS as flag output"""
        return luigi.LocalTarget(f"data/finished_data/_SUCCESS{self.lower}-{self.upper}")

    def requires(self) -> list:
        """Hopefully runs the downloader in parallel if I kick off the final task properly"""
        # TODO ACTually, need this to kick off only the downloader for the lower & upper range of this
        # task. The next task will kick off all of the parallelization.

        # chunk = self.n_urls // self.n_workers
        # assert self.n_urls % self.n_workers == 0  # if this isn't an integer, I want an error
        # #[data_downloader.ImageDownloader(lower=i, upper=i + chunk) for i in range(0, self.n_urls, chunk)]

        return data_downloader.ImageDownloader(lower=self.lower, upper=self.upper)

    def run(self):
        """Big workhorse."""
        # TODO: only read in the rows this instance of the task is responsible for
        df = pd.read_csv(os.path.join("data", self.tabular_path))
        rows = df.iloc[self.lower : self.upper]

        for i, observation in row.iterrows:
            data_cube = preprocessing_utils.get_data_cube(observation)
            file_name = os.path.join("data", "finished_data", salted.get_salted_version(self) + ".npy")

            # This or just the save numpy thing? Would probably want to just use that...
            # Need to be more hands on with that to do atomically
            with open(file_name, mode="w") as file:
                file.write(data_cube)

        # writing the success file
        with self.output().open("w") as outfile:
            outfile.write(f"Success: {self.lower}-{self.upper}")


class GenTrainTestData(luigi.Task):
    """Skeleton code for possibly how I'll do the test & train split and prep before training"""

    test_directory = os.path.join("data", "test")
    train_directory = os.path.join("data", "train")
    SPLIT = 0.8  # fraction of dataset that is trained on

    def output(self):
        # Oh not going to work, outputs are entire directories? Need to resolve. Also with the Salted stuff...
        paths = ...
        return {"test": luigi.LocalTarget(self.test_directory), "train": luigi.LocalTarget(self.train_directory)}

    def requires(self):
        """ """
        return final_project.preprocessing.ToNumpy()
