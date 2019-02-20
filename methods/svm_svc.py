from methods.hyperparameters import *
from sklearn.svm import SVC


def get_name():
    return "SVM SVC"


def get_model_class():
    return SVC


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    C = FloatHyperparameter("C", 0.03125, 32768, default=1.0)
    # No linear kernel here, because we have liblinear
    kernel = CategoricalHyperparameter(name="kernel",
                                       choices=["rbf", "poly", "sigmoid"],
                                       default="rbf")
    degree = IntegerHyperparameter("degree", 1, 5, default=3)
    gamma = FloatHyperparameter("gamma", 3.0517578125e-05, 8, default=0.1)
    # TODO this is totally ad-hoc
    coef0 = FloatHyperparameter("coef0", -1, 1, default=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                          default="True")
    tol = FloatHyperparameter("tol", 1e-5, 1e-1, default=1e-4)
    # cache size is not a hyperparameter, but an argument to the program!
    max_iter = Constant("max_iter", -1)

    hs.add_hyperparameter(C)
    hs.add_hyperparameter(kernel)
    hs.add_hyperparameter(degree)
    hs.add_hyperparameter(gamma)
    hs.add_hyperparameter(coef0)
    hs.add_hyperparameter(shrinking)
    hs.add_hyperparameter(tol)
    hs.add_hyperparameter(max_iter)
        
    return hs
