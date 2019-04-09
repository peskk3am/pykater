from methods.hyperparameters import *
from sklearn.linear_model.stochastic_gradient import SGDClassifier


def get_name():
    return "Stochastic gradient classifier"


def get_model_class():
    return SGDClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    loss = hs.add_hyperparameter(CategoricalHyperparameter("sgd__loss",
        ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        default="log"))
    penalty = hs.add_hyperparameter(CategoricalHyperparameter(
        "sgd__penalty", ["l1", "l2", "elasticnet"], default="l2"))
    alpha = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__alpha", 10e-7, 1e-1, default=0.0001))
    l1_ratio = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__l1_ratio", 1e-9, 1, default=0.15))
    # fit_intercept = hs.add_hyperparameter(Constant("sgd__fit_intercept", True))
    
    max_iter = hs.add_hyperparameter(IntegerHyperparameter(
        "sgd__max_iter", 5, 1000, default=20))
    epsilon = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__epsilon", 1e-5, 1e-1, default=1e-4))
    learning_rate = hs.add_hyperparameter(CategoricalHyperparameter(
        "sgd__learning_rate", ["optimal", "invscaling", "constant"],
        default="optimal"))
    eta0 = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__eta0", 10**-7, 0.1, default=0.01))
    power_t = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__power_t", 1e-5, 1, default=0.25))
    average = hs.add_hyperparameter(CategoricalHyperparameter(
        "sgd__average", [False, True], default=False))

    return hs
