from methods.hyperparameters import *
from sklearn.neighbors import KNeighborsClassifier

import math 

def get_name():
    return "K Neighbors"


def get_model_class():    
    return KNeighborsClassifier


def get_hyperparameter_search_space(n_samples = None):
    cv = 15 # TODO !!!
    
    print("---------------------n_samples, samples - 15 % cv:",  n_samples, math.floor(n_samples*(1-cv/100)))
    if n_samples:
        _upper = min(100, math.floor(n_samples*(1-cv/100)))

    hs = HyperparameterSpace()

    hs.add_hyperparameter(IntegerHyperparameter(name="knn__n_neighbors", lower=1, upper=_upper, default=5))
    hs.add_hyperparameter(CategoricalHyperparameter(name="knn__weights", choices=["uniform", "distance"], default="uniform"))
    hs.add_hyperparameter(CategoricalHyperparameter(name="knn__p", choices=[1, 2], default=2)) 
    hs.add_hyperparameter(CategoricalHyperparameter(name="knn__algorithm", choices=["ball_tree", "kd_tree", "brute"], default="ball_tree"))
    
    return hs