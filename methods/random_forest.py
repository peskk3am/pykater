from methods.hyperparameters import *
from sklearn.ensemble import RandomForestClassifier


def get_name():
    return "Random Forest"


def get_model_class():
    return RandomForestClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    hs.add_hyperparameter(CategoricalHyperparameter(
        "criterion", ["gini", "entropy"], default="gini"))
    #hs.add_hyperparameter(FloatHyperparameter(
    #    "max_features", 0.5, 5, default=1))
    hs.add_hyperparameter(Constant("max_depth", None))
    hs.add_hyperparameter(IntegerHyperparameter(
        "min_samples_split", 2, 20, default=2))
    hs.add_hyperparameter(IntegerHyperparameter(
        "min_samples_leaf", 1, 20, default=1))
    hs.add_hyperparameter(Constant("min_weight_fraction_leaf", 0.))
    hs.add_hyperparameter(Constant("max_leaf_nodes", None))
    hs.add_hyperparameter(CategoricalHyperparameter(
        "bootstrap", ["True", "False"], default="True"))
    # hs.add_hyperparameter(Constant("random_state", 88))
        
    return hs
