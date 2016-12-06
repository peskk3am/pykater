from methods.hyperparameters import *
from sklearn.naive_bayes import GaussianNB


def get_name():
    return "Gaussian NB"


def get_model_class():
    return GaussianNB


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    return hs