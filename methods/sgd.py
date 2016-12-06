from methods.hyperparameters import *
from sklearn.linear_model.stochastic_gradient import SGDClassifier


def get_name():
    return "Stochastic gradient classifier"


def get_model_class():
    return SGDClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    loss = hs.add_hyperparameter(CategoricalHyperparameter("loss",
        ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        default="log"))
    penalty = hs.add_hyperparameter(CategoricalHyperparameter(
        "penalty", ["l1", "l2", "elasticnet"], default="l2"))
    alpha = hs.add_hyperparameter(FloatHyperparameter(
        "alpha", 10e-7, 1e-1, default=0.0001))
    l1_ratio = hs.add_hyperparameter(FloatHyperparameter(
        "l1_ratio", 1e-9, 1, default=0.15))
    #fit_intercept = cs.add_hyperparameter(UnParametrizedHyperparameter(
    #    "fit_intercept", "True"))
    fit_intercept = hs.add_hyperparameter(Constant("fit_intercept", "True"))
    
    n_iter = hs.add_hyperparameter(IntegerHyperparameter(
        "n_iter", 5, 1000, default=20))
    epsilon = hs.add_hyperparameter(FloatHyperparameter(
        "epsilon", 1e-5, 1e-1, default=1e-4))
    learning_rate = hs.add_hyperparameter(CategoricalHyperparameter(
        "learning_rate", ["optimal", "invscaling", "constant"],
        default="optimal"))
    eta0 = hs.add_hyperparameter(FloatHyperparameter(
        "eta0", 10**-7, 0.1, default=0.01))
    power_t = hs.add_hyperparameter(FloatHyperparameter(
        "power_t", 1e-5, 1, default=0.25))
    average = hs.add_hyperparameter(CategoricalHyperparameter(
        "average", ["False", "True"], default="False"))

#    # TODO add passive/aggressive here, although not properly documented?
#    elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
#    epsilon_condition = EqualsCondition(epsilon, loss, "modified_huber")
#    # eta0 seems to be always active according to the source code; when
#    # learning_rate is set to optimial, eta0 is the starting value:
#    # https://github.com/scikit-learn/scikit-learn/blob/0.15.X/sklearn/linear_model/sgd_fast.pyx
#    #eta0_and_inv = EqualsCondition(eta0, learning_rate, "invscaling")
#    #eta0_and_constant = EqualsCondition(eta0, learning_rate, "constant")
#    #eta0_condition = OrConjunction(eta0_and_inv, eta0_and_constant)
#    power_t_condition = EqualsCondition(power_t, learning_rate, "invscaling")

    return hs
