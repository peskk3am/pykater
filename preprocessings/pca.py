from methods.hyperparameters import *
from sklearn.decomposition import PCA


def get_name():
    return "PCA"


def get_model_class():
    return PCA


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    hs.add_hyperparameter(IntegerHyperparameter("pca__n_components", 1, 20, default=None))
    hs.add_hyperparameter(CategoricalHyperparameter("pca__whiten",[True, False], default=False))

    hs.add_hyperparameter(CategoricalHyperparameter(
        "pca__svd_solver", ["auto", "full", "arpack", "randomized"], default="auto"))
            
    return hs
