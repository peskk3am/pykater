from methods.hyperparameters import *
from sklearn.preprocessing import StandardScaler


def get_name():
    return "StandardScaler"


def get_model_class():
    return StandardScaler


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()
    
    hs.add_hyperparameter(CategoricalHyperparameter("scale__with_mean", [True, False], default=True))
    hs.add_hyperparameter(CategoricalHyperparameter("scale__with_std", [True, False], default=True))
            
    return hs
