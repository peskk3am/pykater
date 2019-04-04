from methods.hyperparameters import *
from sklearn.preprocessing import Normalizer


def get_name():
    return "Normalizer"


def get_model_class():
    return Normalizer


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    hs.add_hyperparameter(CategoricalHyperparameter("normalize__norm", ["l1", "l2", "max"], default="l2"))
            
    return hs
