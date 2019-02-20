from methods.hyperparameters import *
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier


def get_name():
    return "Passive aggressive"


def get_model_class():
    return PassiveAggressiveClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()
    
    loss = CategoricalHyperparameter("loss",
                                     ["hinge", "squared_hinge"],
                                     default="hinge")
    fit_intercept = Constant("fit_intercept", "True")
    n_iter = IntegerHyperparameter("n_iter", 5, 1000, default=20)
    C = FloatHyperparameter("C", 1e-5, 10, 1)
        
    hs.add_hyperparameter(loss)
    hs.add_hyperparameter(fit_intercept)
    hs.add_hyperparameter(n_iter)
    hs.add_hyperparameter(C)
        
    return hs
