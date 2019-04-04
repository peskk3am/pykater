from methods.hyperparameters import *
from sklearn.naive_bayes import BernoulliNB


def get_name():
    return "Bernoulli NB"


def get_model_class():
    return BernoulliNB


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    # the smoothing parameter is a non-negative float
    # I will limit it to 1000 and put it on a logarithmic scale. (SF)
    # Please adjust that, if you know a proper range, this is just a guess.
    alpha = FloatHyperparameter(name="bernoulli_nb__alpha", lower=1e-2, upper=100, default=1)

    fit_prior = CategoricalHyperparameter(name="bernoulli_nb__fit_prior",
                                          choices=["True", "False"],
                                          default="True")
    
    hs.add_hyperparameter(alpha)
    hs.add_hyperparameter(fit_prior)      

    return hs
