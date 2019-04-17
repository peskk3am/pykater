from methods.hyperparameters import *
from sklearn.tree import DecisionTreeClassifier

def get_name():
    return "Decision Tree"


def get_model_class():
    return DecisionTreeClassifier


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    hs.add_hyperparameter(CategoricalHyperparameter("decision_tree__criterion", ["gini", "entropy"], default="gini"))
    #hs.add_hyperparameter(Constant("decision_tree__splitter", "best"))
    #hs.add_hyperparameter(Constant('decision_tree__max_features', None))
    hs.add_hyperparameter(FloatHyperparameter('decision_tree__max_depth', 0.01, 100.0, default=0.5))
    hs.add_hyperparameter(IntegerHyperparameter("decision_tree__min_samples_split", 2, 20, default=2))
    hs.add_hyperparameter(IntegerHyperparameter("decision_tree__min_samples_leaf", 1, 20, default=1))
    #hs.add_hyperparameter(Constant("decision_tree__min_weight_fraction_leaf", 0.0))
    #hs.add_hyperparameter(UnParametrizedHyperparameter("decision_tree__max_leaf_nodes", "None"))

    return hs