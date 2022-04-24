import luigi
import keras
import hydra

from final_project import preprocessing


@hydra.config("cfg/training") # something like that
class TrainModel(luigi.Task):
    salt = pass

    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def requires(self):
        return preprocessing.GenTrainData()

    def output(self):
        return luigi.LocalTarget(f'model_{self.salt}.something')
