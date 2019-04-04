from methods.hyperparameters import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def get_name():
    return "Quadratic discriminant analysis"


def get_model_class():
    return QuadraticDiscriminantAnalysis


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    reg_param = FloatHyperparameter('qda__reg_param', 0.0, 10.0,
                                           default=0.5)
    tol = FloatHyperparameter('qda__tol', 0.0, 1.0,
                                           default=1.0e-4)
    
    hs.add_hyperparameter(reg_param)    
    hs.add_hyperparameter(tol)    

    return hs
