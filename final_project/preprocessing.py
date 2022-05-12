import os
from os.path import join
import zipfile
import glob

import luigi
import numpy as np
import omegaconf
import pandas as pd

from final_project import data_downloader
from final_project import preprocessing_utils
from final_project import salted

# path issues with sphinx and the relative paths for running as a module, as intended when I wrote them
abs_path = os.path.dirname(__file__)


class Preprocessing(luigi.Task):
    """Takes the zipped image files and ends up with the finished, saved datacubes in .npy format"""

    __version__ = "0.1.0"

    # Grabs the tabular data's name, want to be agnostic to future changes to that file
    tabular_path = omegaconf.OmegaConf.load(join(abs_path, "conf", "aws_paths.yaml"))["tabular_data"]
    local_paths = omegaconf.OmegaConf.load(join(abs_path, "conf", "local_paths.yaml"))
    processed_path = join(abs_path, "..", local_paths["data"], local_paths["processed"])

    # range of downloaded files to process = [lower, upper]
    lower = luigi.IntParameter()
    upper = luigi.IntParameter()

    def output(self) -> luigi.LocalTarget:
        """The 'I'm done' file for this task is a _SUCCESS flag file"""
        return luigi.LocalTarget(
            join(self.processed_path, f"_SUCCESS{self.lower}-{self.upper}-{salted.get_salted_version(self)}")
        )

    def requires(self) -> list:
        """
        Only kicks off one instance of ImageDownloader, so the number of instances of the Preprocessing task is the
        same as the number of instances of ImageDownloader.
        """
        return data_downloader.ImageDownloader(lower=self.lower * 5, upper=self.upper * 5)

    def run(self) -> None:
        """
        Currently filled with a lot of log writing. Opens the downloaded tabular data and creates
        one data cube for each galaxy in the catalog from the 5 fits images we downloaded in ImageDownloader.
        Calls preprocessing_utils.get_data_cube if finished datacube npy doesn't exist already.

        If something fails with a corrupted file or one required file was not downloaded, it logs it and continues with
        the processing.
        """
        df = pd.read_csv(join(abs_path, "..", self.local_paths["data"], self.tabular_path))
        rows = df.iloc[self.lower : self.upper]

        # logging
        logs = ""
        logs_not_found = ""
        logs_bad_data = ""

        for i, observation in rows.iterrows():
            file_name = join(self.processed_path, f"{observation.objID}-{salted.get_salted_version(self)}.npy")

            try:
                # for debugging, want to see that it tried every file it was assigned
                logs = logs + "\n" + str(i)

                # this glob will give me all of the files that match the obsveration ID in the processed directory
                # it's our skip if this already exists function, but agnostic to the salt for now
                if glob.glob(file_name[:-15] + "*") == []:
                    data_cube = preprocessing_utils.get_data_cube(observation)
                    # can I improve this?
                    np.save(file=file_name, arr=data_cube)
                    print("Done with ", file_name)
                else:
                    print("Already finished: ", file_name)
            except FileNotFoundError:
                logs_not_found += "\n" + file_name
            except OSError:
                logs_bad_data += "\n" + file_name

        # writing a log of the files that were corrupted
        with open(f"debug/bad_data{self.lower}.txt", mode="w") as log:
            log.write(logs_bad_data)

        # writing a log of the files that weren't found
        with open(f"debug/log_file{self.lower}.txt", mode="w") as log:
            log.write(logs_not_found)

        # writing the log file
        with open(f"debug/debugging{self.lower}-{self.upper}.txt", mode="w") as debug:
            debug.write(logs)

        # writing the success file
        with self.output().open("w") as outfile:
            outfile.write(f"Success: {self.lower}-{self.upper}")


# class GenTrainTestData(luigi.Task):
#     """Skeleton code for possibly how I'll do the test & train split and prep before training"""

#     local_paths = omegaconf.OmegaConf.load(join(abs_path, "conf", "local_paths.yaml"))

#     test_directory = join(abs_path, local_paths["data"], local_paths["test"])
#     train_directory = join(abs_path, local_paths["data"], local_paths["train"])
#     SPLIT = 0.8  # fraction of dataset that is trained on

#     def output(self):
#         # Oh not going to work, outputs are entire directories? Need to resolve. Also with the Salted stuff...
#         paths = ...
#         return {"test": luigi.LocalTarget(self.test_directory), "train": luigi.LocalTarget(self.train_directory)}

#     def requires(self):
#         """ """
#         return final_project.preprocessing.ToNumpy()
