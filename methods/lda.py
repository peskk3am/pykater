from methods.hyperparameters import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_name():
    return "Linear discriminant analysis"


def get_model_class():
    return LinearDiscriminantAnalysis


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    shrinkage = hs.add_hyperparameter(CategoricalHyperparameter(
        "shrinkage", ["None", "auto", "manual"], default="None"))
    shrinkage_factor = hs.add_hyperparameter(FloatHyperparameter(
        "shrinkage_factor", 0., 1., 0.5))
    n_components = hs.add_hyperparameter(IntegerHyperparameter(
        'n_components', 1, 250, default=10))
    tol = hs.add_hyperparameter(FloatHyperparameter(
        "tol", 1e-5, 1e-1, default=1e-4))
     
    return hs
