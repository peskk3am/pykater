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
        "sgd__alpha", 0.0006534373590216062, 0.1, default=0.0001))
    l1_ratio = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__l1_ratio", 0.00847809819764737, 0.9904732045803555, default=0.15))
    fit_intercept = hs.add_hyperparameter(CategoricalHyperparameter(
        "sgd__fit_intercept", [True, False], True))    
    max_iter = hs.add_hyperparameter(IntegerHyperparameter(
        "sgd__max_iter", 50, 999, default=5))
    epsilon = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__epsilon",  1e-05, 0.1, default=1e-4))
    learning_rate = hs.add_hyperparameter(CategoricalHyperparameter(
        "sgd__learning_rate", ["optimal", "invscaling", "constant", "adaptive"],
        default="optimal"))
    eta0 = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__eta0", 0.0053481428725508096, 0.9993790130712993, default=0.0))
    power_t = hs.add_hyperparameter(FloatHyperparameter(
        "sgd__power_t", 0.00046061890638235376, 0.9937186575441405, default=0.5))
    average = hs.add_hyperparameter(CategoricalHyperparameter(
        "sgd__average", [False, True], default=False))
    average = hs.add_hyperparameter(CategoricalHyperparameter(
        "sgd__shuffle", [False, True], default=True))

    return hs
