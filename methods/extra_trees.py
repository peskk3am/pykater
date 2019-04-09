from methods.hyperparameters import *
from sklearn.ensemble import ExtraTreesClassifier


def get_name():
    return "Extra trees"


def get_model_class():
    return ExtraTreesClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    n_estimators = hs.add_hyperparameter(IntegerHyperparameter("extra_trees__n_estimators", 1, 100, default=10))
    criterion = hs.add_hyperparameter(CategoricalHyperparameter(
        "extra_trees__criterion", ["gini", "entropy"], default="gini"))
    max_features = hs.add_hyperparameter(FloatHyperparameter(
        "extra_trees__max_features", 0.5, 5.0, default=1.0))

    max_depth = hs.add_hyperparameter(IntegerHyperparameter("extra_trees__max_depth", 1, 100, default=None))

    min_samples_split = hs.add_hyperparameter(IntegerHyperparameter(
        "extra_trees__min_samples_split", 2, 20, default=2))
    min_samples_leaf = hs.add_hyperparameter(IntegerHyperparameter(
        "extra_trees__min_samples_leaf", 1, 20, default=1))
        
    bootstrap = hs.add_hyperparameter(CategoricalHyperparameter(
        "extra_trees__bootstrap", [True, False], default=False))
      
    return hs
