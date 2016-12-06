from methods.hyperparameters import *
from sklearn.multiclass import OneVsRestClassifier


def get_name():
    return "One vs rest (multinomial nb)"


def get_model_class():
    return OneVsRestClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()
    
    # the smoothing parameter is a non-negative float
    # I will limit it to 100 and put it on a logarithmic scale. (SF)
    # Please adjust that, if you know a proper range, this is just a guess.
    alpha = hs.add_hyperparameter(FloatHyperparameter(name="alpha", 
                                  lower=1e-2, upper=100, default=1))

    fit_prior = hs.add_hyperparameter(CategoricalHyperparameter(
                                          name="fit_prior",
                                          choices=["True", "False"],
                                          default="True"))    

    return hs
