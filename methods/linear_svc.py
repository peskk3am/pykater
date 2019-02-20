from methods.hyperparameters import *
from sklearn.svm import LinearSVC


def get_name():
    return "Linear SVC"


def get_model_class():
    return LinearSVC


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    penalty = hs.add_hyperparameter(CategoricalHyperparameter(
        "penalty", ["l1", "l2"], default="l2"))
    loss = hs.add_hyperparameter(CategoricalHyperparameter(
        "loss", ["hinge", "squared_hinge"], default="squared_hinge"))
    dual = hs.add_hyperparameter(Constant("dual", "False"))
    # This is set ad-hoc
    tol = hs.add_hyperparameter(FloatHyperparameter(
        "tol", 1e-5, 1e-1, default=1e-4))
    C = hs.add_hyperparameter(FloatHyperparameter(
        "C", 0.03125, 32768, default=1.0))
    multi_class = hs.add_hyperparameter(Constant("multi_class", "ovr"))
    # These are set ad-hoc
    fit_intercept = hs.add_hyperparameter(Constant("fit_intercept", "True"))
    intercept_scaling = hs.add_hyperparameter(Constant(
        "intercept_scaling", 1))

#    penalty_and_loss = ForbiddenAndConjunction(
#        ForbiddenEqualsClause(penalty, "l1"),
#        ForbiddenEqualsClause(loss, "hinge")
#    )
#    constant_penalty_and_loss = ForbiddenAndConjunction(
#        ForbiddenEqualsClause(dual, "False"),
#        ForbiddenEqualsClause(penalty, "l2"),
#        ForbiddenEqualsClause(loss, "hinge")
#    )
#    penalty_and_dual = ForbiddenAndConjunction(
#        ForbiddenEqualsClause(dual, "False"),
#        ForbiddenEqualsClause(penalty, "l1")
#    )
#    cs.add_forbidden_clause(penalty_and_loss)
#    cs.add_forbidden_clause(constant_penalty_and_loss)
#    cs.add_forbidden_clause(penalty_and_dual)

    return hs
