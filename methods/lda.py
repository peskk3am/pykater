from methods.hyperparameters import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_name():
    return "Linear discriminant analysis"


def get_model_class():
    return LinearDiscriminantAnalysis


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    solver = hs.add_hyperparameter(CategoricalHyperparameter(
        "lda__solver", ["svd", "lsqr", "eigen"], default="svd")) 
    shrinkage = hs.add_hyperparameter(CategoricalHyperparameter(
        "lda__shrinkage", [None, "auto", 0.1, 0.5, 0.9], default=None))   # float between 0 and 1: fixed shrinkage parameter.
#    shrinkage_factor = hs.add_hyperparameter(FloatHyperparameter(
#        "lda__shrinkage_factor", 0., 1., 0.5))
    n_components = hs.add_hyperparameter(IntegerHyperparameter(
        'lda__n_components', 1, 250, default=10))
    tol = hs.add_hyperparameter(FloatHyperparameter(
        "lda__tol", 1e-5, 1e-1, default=1e-4))
     
    return hs
