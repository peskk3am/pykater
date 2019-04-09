from methods.hyperparameters import *
from sklearn.svm import LinearSVC


def get_name():
    return "Linear SVC"


def get_model_class():
    return LinearSVC


def get_hyperparameter_search_space():

    hs = HyperparameterSpace()

    penalty = hs.add_hyperparameter(CategoricalHyperparameter(
        "linear_svc__penalty", ["l1", "l2"], default="l2"))
    loss = hs.add_hyperparameter(CategoricalHyperparameter(
        "linear_svc__loss", ["hinge", "squared_hinge"], default="squared_hinge"))
    # This is set ad-hoc
    tol = hs.add_hyperparameter(FloatHyperparameter(
        "linear_svc__tol", 1e-5, 1e-1, default=1e-4))
    C = hs.add_hyperparameter(FloatHyperparameter(
        "linear_svc__C", 0.001, 10000, default=1.0))

#    multi_class = hs.add_hyperparameter(Constant("linear_svc__multi_class", "ovr"))
#    # These are set ad-hoc
#    fit_intercept = hs.add_hyperparameter(Constant("linear_svc__fit_intercept", True))
#    intercept_scaling = hs.add_hyperparameter(Constant(
#        "linear_svc__intercept_scaling", 1))
#
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
