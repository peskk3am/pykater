from methods.hyperparameters import *
from sklearn.ensemble import AdaBoostClassifier

def get_name():
    return "AdaBoost"


def get_model_class():
    return AdaBoostClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    n_estimators = hs.add_hyperparameter(IntegerHyperparameter(
        name="n_estimators", lower=50, upper=500, default=50))
    learning_rate = hs.add_hyperparameter(FloatHyperparameter(
        name="learning_rate", lower=0.01, upper=2, default=0.1))
    algorithm = hs.add_hyperparameter(CategoricalHyperparameter(
        name="algorithm", choices=["SAMME.R", "SAMME"], default="SAMME.R"))
    #max_depth = hs.add_hyperparameter(IntegerHyperparameter(
    #    name="max_depth", lower=1, upper=10, default=1))

    return hs