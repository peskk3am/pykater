from methods.hyperparameters import *
from sklearn.ensemble import RandomForestClassifier


def get_name():
    return "Random Forest"


def get_model_class():
    return RandomForestClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    hs.add_hyperparameter(IntegerHyperparameter(
        "random_forest__n_estimators", 9, 100, default=10))

    hs.add_hyperparameter(CategoricalHyperparameter(
        "random_forest__criterion", ["gini", "entropy"], default="gini"))

    hs.add_hyperparameter(IntegerHyperparameter(
        "random_forest__max_depth", 5, 100, default=None))

    hs.add_hyperparameter(IntegerHyperparameter(
        "random_forest__max_features", 3, 6, default=None))
    
    hs.add_hyperparameter(IntegerHyperparameter(
        "random_forest__min_samples_split", 2, 20, default=2))
        
    hs.add_hyperparameter(IntegerHyperparameter(
        "random_forest__min_samples_leaf", 1, 20, default=1))
        
    hs.add_hyperparameter(FloatHyperparameter("random_forest__min_weight_fraction_leaf", 0.0, 0.13726647541107984, 0.0))
        
    hs.add_hyperparameter(CategoricalHyperparameter(
        "random_forest__bootstrap", [True, False], default=True))
        
    return hs
