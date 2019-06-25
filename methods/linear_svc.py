from methods.hyperparameters import *
from sklearn.svm import LinearSVC


def get_name():
    return "Linear SVC"


def get_model_class():
    return LinearSVC


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    penalty = hs.add_hyperparameter(CategoricalHyperparameter(
        "linear_svc__penalty", ["l1", "l2"], default="l2"))
    loss = hs.add_hyperparameter(CategoricalHyperparameter(
        "linear_svc__loss", ["hinge", "squared_hinge"], default="squared_hinge"))    
    tol = hs.add_hyperparameter(FloatHyperparameter(
        "linear_svc__tol", 1e-5, 1e-1, default=1e-4))
    C = hs.add_hyperparameter(FloatHyperparameter(
        "linear_svc__C", 0.001, 10000.0, default=1.0))
    multi_class = hs.add_hyperparameter(CategoricalHyperparameter(
        "linear_svc__multi_class", ["ovr", "crammer_singer"], default="ovr"))
    fit_intercept = hs.add_hyperparameter(CategoricalHyperparameter("linear_svc__fit_intercept", [True, False], default=True))
    intercept_scaling = hs.add_hyperparameter(FloatHyperparameter(
        "linear_svc__intercept_scaling", lower=0, upper=10, default=1))

    hs.add_hyperparameter(IntegerHyperparameter(
        "linear_svc__max_iter", 10, 100, default=1000))

    return hs
