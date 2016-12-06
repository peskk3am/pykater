from methods.hyperparameters import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def get_name():
    return "Quadratic discriminant analysis"


def get_model_class():
    return QuadraticDiscriminantAnalysis


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    reg_param = FloatHyperparameter('reg_param', 0.0, 10.0,
                                           default=0.5)
    hs.add_hyperparameter(reg_param)    

    return hs
