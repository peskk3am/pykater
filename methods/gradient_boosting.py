from methods.hyperparameters import *
from sklearn.ensemble import GradientBoostingClassifier


def get_name():
    return "Gradient boosting"


def get_model_class():
    return GradientBoostingClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    loss = hs.add_hyperparameter(CategoricalHyperparameter(
        "gradient_boosting__loss", ["deviance", "exponential"], default="deviance"))
    learning_rate = hs.add_hyperparameter(FloatHyperparameter(
        name="gradient_boosting__learning_rate", lower=0.01, upper=1.0, default=0.1))
    n_estimators = hs.add_hyperparameter(IntegerHyperparameter
        ("gradient_boosting__n_estimators", 50, 500, default=100))
    subsample = hs.add_hyperparameter(FloatHyperparameter(
            name="gradient_boosting__subsample", lower=0.01, upper=1.0, default=1.0))
    criterion = hs.add_hyperparameter(CategoricalHyperparameter(
        "gradient_boosting__criterion", ["friedman_mse", "mse", "mae"], default="friedman_mse"))
    min_samples_split = hs.add_hyperparameter(IntegerHyperparameter(
        name="gradient_boosting__min_samples_split", lower=2, upper=20, default=2))
    min_weight_fraction_leaf = hs.add_hyperparameter(                              
        FloatHyperparameter("gradient_boosting__min_weight_fraction_leaf", lower=0, upper=1, default=0.0))      
    max_depth = hs.add_hyperparameter(IntegerHyperparameter(
        name="gradient_boosting__max_depth", lower=1, upper=10, default=3))
    min_samples_leaf = hs.add_hyperparameter(IntegerHyperparameter(
        name="gradient_boosting__min_samples_leaf", lower=1, upper=20, default=1))
    max_features = hs.add_hyperparameter(IntegerHyperparameter(
        "gradient_boosting__max_features", 1, 20, default=None))
      
    return hs
