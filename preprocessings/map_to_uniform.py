from methods.hyperparameters import *
from sklearn.preprocessing import QuantileTransformer

'''
    This method transforms the features to follow a uniform or a normal distribution. 
'''

def get_name():
    return "QuantileTransformer"


def get_model_class():
    return QuantileTransformer


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    hs.add_hyperparameter(IntegerHyperparameter("map_to_uniform__n_quantiles", 100, 5000, default=1000))
    hs.add_hyperparameter(CategoricalHyperparameter("map_to_uniform__output_distribution", ["normal", "uniform"], default="uniform"))
    hs.add_hyperparameter(CategoricalHyperparameter("map_to_uniform__ignore_implicit_zeros", [True, False], default=False))
    
    ''' subsample : int, optional (default=1e5)
        Maximum number of samples used to estimate the quantiles for computational efficiency.
    '''        
    hs.add_hyperparameter(IntegerHyperparameter("map_to_uniform__subsample", 1000, 1e5, default=1e5))
    
    return hs
  