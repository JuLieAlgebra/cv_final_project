import luigi
import os


class GenTrainTestData(luigi.Task):
    test_directory = os.path.join("data", "test")
    train_directory = os.path.join("data", "train")
    SPLIT = 0.8  # fraction of dataset that is trained on

    def output(self):
        paths = ...
        return {"test": luigi.LocalTarget(self.test_directory), "train": luigi.LocalTarget(self.train_directory)}

    def requires(self):
        """ """
        return final_project.preprocessing.ToNumpy()
