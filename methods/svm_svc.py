from methods.hyperparameters import *
from sklearn.svm import SVC


def get_name():
    return "SVM SVC"


def get_model_class():
    return SVC


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    C = FloatHyperparameter("svm_svc__C", 0.001, 10000.0, default=1.0)
    # No linear kernel here, because we have liblinear
    kernel = CategoricalHyperparameter(name="svm_svc__kernel",
                                       choices=["rbf", "poly", "sigmoid"],
                                       default="rbf")
    degree = IntegerHyperparameter("svm_svc__degree", 1, 5, default=3)
    gamma = FloatHyperparameter("svm_svc__gamma", 3.0e-05, 10.0, default=0.1)
    coef0 = FloatHyperparameter("svm_svc__coef0", -1.0, 1.0, default=0.0)    
    shrinking = CategoricalHyperparameter("svm_svc__shrinking", [True, False], default=True)
    tol = FloatHyperparameter("svm_svc__tol", 1e-5, 1e-1, default=1e-3)        
    decision_function_shape = CategoricalHyperparameter("svm_svc__decision_function_shape", ["ovo", "ovr"], default="ovr")

    hs.add_hyperparameter(C)
    hs.add_hyperparameter(kernel)
    hs.add_hyperparameter(degree)
    hs.add_hyperparameter(gamma)
    hs.add_hyperparameter(coef0)
    hs.add_hyperparameter(shrinking)
    hs.add_hyperparameter(tol)
    hs.add_hyperparameter(decision_function_shape)
        
    return hs
