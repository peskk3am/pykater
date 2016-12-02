from methods.hyperparameters import *
from sklearn.neighbors import KNeighborsClassifier


def get_name():
    return "K Neighbors"


def get_model_class():
    return KNeighborsClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    hs.add_hyperparameter(IntegerHyperparameter(name="n_neighbors", lower=1, upper=50, default=1))
    hs.add_hyperparameter(CategoricalHyperparameter(name="weights", choices=["uniform", "distance"], default="uniform"))
    hs.add_hyperparameter(CategoricalHyperparameter(name="p", choices=[1, 2], default=2)) 

    return hs