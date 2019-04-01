from methods.hyperparameters import *
from sklearn.ensemble import ExtraTreesClassifier


def get_name():
    return "Extra trees"


def get_model_class():
    return ExtraTreesClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    n_estimators = hs.add_hyperparameter(Constant("n_estimators", 100))
    criterion = hs.add_hyperparameter(CategoricalHyperparameter(
        "criterion", ["gini", "entropy"], default="gini"))
    max_features = hs.add_hyperparameter(FloatHyperparameter(
        "max_features", 0.5, 5, default=1))

    max_depth = hs.add_hyperparameter(Constant(name="max_depth", value=None))

    min_samples_split = hs.add_hyperparameter(IntegerHyperparameter(
        "min_samples_split", 2, 20, default=2))
    min_samples_leaf = hs.add_hyperparameter(IntegerHyperparameter(
        "min_samples_leaf", 1, 20, default=1))
    min_weight_fraction_leaf = hs.add_hyperparameter(Constant(
        'min_weight_fraction_leaf', 0.))

    bootstrap = hs.add_hyperparameter(CategoricalHyperparameter(
        "bootstrap", [True, False], default=False))
      
    return hs
