from methods.hyperparameters import *
from sklearn.ensemble import AdaBoostClassifier

def get_name():
    return "AdaBoost"


def get_model_class():
    return AdaBoostClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    n_estimators = hs.add_hyperparameter(IntegerHyperparameter(
        name="adaboost__n_estimators", lower=193, upper=500, default=50))
    learning_rate = hs.add_hyperparameter(FloatHyperparameter(
        name="adaboost__learning_rate", lower=0.4349709832652311, upper=1.234792157524018, default=1))
    algorithm = hs.add_hyperparameter(CategoricalHyperparameter(
        name="adaboost__algorithm", choices=["SAMME.R", "SAMME"], default="SAMME.R"))

    return hs