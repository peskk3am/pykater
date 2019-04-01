from methods.hyperparameters import *
from sklearn.ensemble import GradientBoostingClassifier


def get_name():
    return "Gradient boosting"


def get_model_class():
    return GradientBoostingClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    loss = hs.add_hyperparameter(Constant("loss", "deviance"))
    learning_rate = hs.add_hyperparameter(FloatHyperparameter(
        name="learning_rate", lower=0.01, upper=1, default=0.1))
    n_estimators = hs.add_hyperparameter(IntegerHyperparameter
        ("n_estimators", 50, 500, default=100))
    max_depth = hs.add_hyperparameter(IntegerHyperparameter(
        name="max_depth", lower=1, upper=10, default=3))
    min_samples_split = hs.add_hyperparameter(IntegerHyperparameter(
        name="min_samples_split", lower=2, upper=20, default=2))
    min_samples_leaf = hs.add_hyperparameter(IntegerHyperparameter(
        name="min_samples_leaf", lower=1, upper=20, default=1))
    min_weight_fraction_leaf = hs.add_hyperparameter(
        Constant("min_weight_fraction_leaf", 0.))
    subsample = hs.add_hyperparameter(FloatHyperparameter(
            name="subsample", lower=0.01, upper=1.0, default=1.0))
    max_features = hs.add_hyperparameter(IntegerHyperparameter(
        "max_features", 1, 10, default=1))
    max_leaf_nodes = hs.add_hyperparameter(Constant(
        name="max_leaf_nodes", value=None))
      
    return hs
