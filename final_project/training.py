import luigi
import keras
import omegaconf

from final_project import preprocessing
from final_project import salted


class TrainModel(luigi.Task):
    # Two config files to avoid code changes
    conf_path = omegaconf.OmegaConf.load("final_project/conf/train_configs.yaml")
    params = omegaconf.OmegaConf.load(conf_path)

    lr = luigi.Parameter(params["lr"])
    ...  # some repeated code in order to have the parameters become part of the salt
    model = Model(params)

    def train(self):
        ...

    def predict(self):
        ...

    def requires(self):
        return  # preprocessing.GenTrainData()

    def output(self):
        """trained model with checkpoints....."""
        return salted.SaltedOutput(self.__class__.__name__)  # something like this


class Model:
    def __init__(self, params: dict):
        ...
        # params = omegaconf.OmegaConf.load("final_project/conf/train.yaml")
