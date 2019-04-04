from methods.hyperparameters import *
from sklearn.preprocessing import PowerTransformer

'''
    Apply a power transform featurewise to make data more Gaussian-like. 
'''

def get_name():
    return "PowerTransformer"


def get_model_class():
    return PowerTransformer


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    hs.add_hyperparameter(CategoricalHyperparameter("map_to_gaussian__method", ["yeo-johnson", "box-cox"], default="yeo-johnson"))
    hs.add_hyperparameter(CategoricalHyperparameter("map_to_gaussian__standardize", [True, False], default=True))
        
    return hs
  