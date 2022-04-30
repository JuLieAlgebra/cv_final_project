import luigi
import keras
import omegaconf

from final_project import preprocessing
from final_project import salted


class TrainModel(luigi.Task):
    params = omegaconf.OmegaConf.load("final_project/conf/train.yaml")
    lr = luigi.Parameter(params["lr"])
    ...  # some repeated code in order to have the parameters become part of the salt
    model = Model(params)

    def train(self):
        self.model.train()  # ish.

    def predict(self):
        pass

    def requires(self):
        return  # preprocessing.GenTrainData()

    def output(self):
        """trained model with checkpoints....."""
        return salted.SaltedOutput(self.__class__.__name__)  # something like this


class Model:
    def __init__(self, params: dict):
        ...
        # params = omegaconf.OmegaConf.load("final_project/conf/train.yaml")
