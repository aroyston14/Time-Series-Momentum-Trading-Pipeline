# Model definitions, hyperparameter grids and time-series tuning.
#
#   majority      - DummyClassifier(strategy="prior"). Always predicts the training set's majority
#                   class and gives the training base rate as its probability, so log loss and
#                   Brier score are meaningful rather than infinite.
#   logreg_none   - unregularised logistic regression (StandardScaler then LR in a Pipeline)
#   logreg_l1     - L1 (lasso) logistic regression, with C tuned
#   logreg_l2     - L2 (ridge) logistic regression, with C tuned
#   random_forest - RandomForestClassifier on unscaled inputs, since trees do not need scaling
#   xgboost       - xgboost.XGBClassifier on unscaled inputs. If xgboost is not installed and
#                   allow_sklearn_fallback is true, HistGradientBoostingClassifier is used instead
#                   and labelled "xgboost[sklearn-hgb-fallback]" everywhere.
#
# The scaler sits inside the logistic regression Pipeline, so it is only ever fitted on the
# training fold.

# Standard library modules for logging, silencing noisy warnings during tuning, and dataclasses
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from momentum_ml.splits import date_grouped_ts_cv

logger = logging.getLogger(__name__)

# scikit-learn 1.8 deprecated the 'penalty' argument in favour of setting C and l1_ratio,
# so the installed version decides which form is used
_SK_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
_NEW_LR_API = _SK_VERSION >= (1, 8)


# Returns True if the xgboost package can be imported
def xgboost_available():
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


# Builds a LogisticRegression with no penalty, an L1 penalty or an L2 penalty, in a way that works
# on both old and new versions of scikit-learn
def make_logistic(penalty, C=1.0, class_weight="balanced", max_iter=2000, seed=0):
    if penalty not in (None, "l1", "l2"):
        raise ValueError(f"Unsupported penalty {penalty!r}")
    shared_args = {"class_weight": class_weight, "max_iter": max_iter, "random_state": seed}

    if _NEW_LR_API:
        if penalty is None:
            return LogisticRegression(C=np.inf, solver="lbfgs", **shared_args)
        if penalty == "l1":
            return LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", **shared_args)
        return LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", **shared_args)

    if penalty is None:
        return LogisticRegression(penalty=None, solver="lbfgs", **shared_args)
    if penalty == "l1":
        return LogisticRegression(penalty="l1", C=C, solver="liblinear", **shared_args)
    return LogisticRegression(penalty="l2", C=C, solver="lbfgs", **shared_args)


# Puts a StandardScaler in front of the logistic regression
def _lr_pipeline(logistic_model):
    return Pipeline([("scaler", StandardScaler()), ("clf", logistic_model)])


# An unfitted model together with its hyperparameter grid
@dataclass
class ModelSpec:
    name: str
    estimator: BaseEstimator
    grid: dict[str, list[Any]] = field(default_factory=dict)
    family: str = ""
    note: str = ""


# Creates every model that is enabled in the "models" section of the config
def build_model_specs(config):
    seed = int(config["project"]["seed"])
    models_config = config["models"]
    specs = {}

    # Majority class baseline
    if models_config.get("majority", {}).get("enabled", True):
        specs["majority"] = ModelSpec(
            "majority", DummyClassifier(strategy="prior"), family="baseline"
        )

    # The three logistic regressions. Only the penalised ones have a C grid to tune.
    for name, penalty in (("logreg_none", None), ("logreg_l1", "l1"), ("logreg_l2", "l2")):
        model_config = models_config.get(name, {})
        if not model_config.get("enabled", True):
            continue
        logistic_model = make_logistic(
            penalty,
            1.0,
            model_config.get("class_weight", "balanced"),
            int(model_config.get("max_iter", 2000)),
            seed,
        )
        grid = {"clf__C": list(model_config.get("grid", {}).get("C", []))} if penalty else {}
        specs[name] = ModelSpec(name, _lr_pipeline(logistic_model), grid, family="logistic")

    # Random forest
    model_config = models_config.get("random_forest", {})
    if model_config.get("enabled", True):
        forest = RandomForestClassifier(
            class_weight=model_config.get("class_weight", "balanced_subsample"),
            random_state=seed,
            n_jobs=1,
        )
        specs["random_forest"] = ModelSpec(
            "random_forest", forest, dict(model_config.get("grid", {})), family="tree"
        )

    # XGBoost, or the scikit-learn stand-in if xgboost is not installed
    model_config = models_config.get("xgboost", {})
    if model_config.get("enabled", True):
        grid = dict(model_config.get("grid", {}))
        if xgboost_available():
            from xgboost import XGBClassifier

            xgb_model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            )
            specs["xgboost"] = ModelSpec("xgboost", xgb_model, grid, family="tree")

        elif model_config.get("allow_sklearn_fallback", True):

            # Translating the xgboost grid names into their HistGradientBoosting equivalents
            fallback_grid = {}
            if "learning_rate" in grid:
                fallback_grid["learning_rate"] = grid["learning_rate"]
            if "max_depth" in grid:
                fallback_grid["max_depth"] = grid["max_depth"]
            if "n_estimators" in grid:
                fallback_grid["max_iter"] = grid["n_estimators"]
            if "colsample_bytree" in grid:
                fallback_grid["max_features"] = grid["colsample_bytree"]
            if "reg_lambda" in grid:
                fallback_grid["l2_regularization"] = grid["reg_lambda"]

            note = (
                "xgboost not installed: HistGradientBoostingClassifier used as a stand-in "
                "(n_estimators->max_iter, colsample_bytree->max_features, "
                "reg_lambda->l2_regularization; row subsampling and reg_alpha (L1) unsupported)"
            )
            logger.warning(note)
            fallback_model = HistGradientBoostingClassifier(random_state=seed, early_stopping=False)
            name = "xgboost[sklearn-hgb-fallback]"
            specs[name] = ModelSpec(name, fallback_model, fallback_grid, family="tree", note=note)
        else:
            logger.error(
                "xgboost is not installed and allow_sklearn_fallback is false; skipping xgboost"
            )

    if not specs:
        raise ValueError("No models enabled in config.models")
    return specs


# The result of tuning one model on the training set.
# estimator has been refitted on the full training set with the best parameters.
@dataclass
class TunedModel:
    name: str
    estimator: BaseEstimator
    best_params: dict[str, Any]
    cv_score: float | None
    cv_results: pd.DataFrame | None
    spec: ModelSpec


# Grid searches a model using date-grouped TimeSeriesSplit CV on the training data only.
# Models without a grid are just fitted. GridSearchCV(refit=True) refits the returned estimator
# on all of X with the best parameters.
def tune_model(spec, X, y, dates, config):
    tuning_config = config["tuning"]
    if not spec.grid:
        fitted_model = clone(spec.estimator).fit(X, y)
        return TunedModel(spec.name, fitted_model, {}, None, None, spec)

    gap = int(config["split"].get("purge_days") or config["target"]["horizon"])
    cv_folds = date_grouped_ts_cv(dates, int(tuning_config["cv_splits"]), gap=gap)
    search = GridSearchCV(
        clone(spec.estimator),
        spec.grid,
        scoring=tuning_config.get("scoring", "neg_log_loss"),
        cv=cv_folds,
        n_jobs=tuning_config.get("n_jobs", -1),
        refit=True,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
        search.fit(X, y)

    # Only the parameter columns and the test scores are kept from the CV results
    cv_results = pd.DataFrame(search.cv_results_)
    score_columns = ("mean_test_score", "std_test_score", "rank_test_score")
    kept_columns = [
        column
        for column in cv_results.columns
        if column.startswith("param_") or column in score_columns
    ]
    logger.info(
        "%s: best CV %s = %.5f with %s",
        spec.name,
        tuning_config.get("scoring"),
        search.best_score_,
        search.best_params_,
    )
    return TunedModel(
        spec.name,
        search.best_estimator_,
        dict(search.best_params_),
        float(search.best_score_),
        cv_results[kept_columns],
        spec,
    )


# Clones the spec's estimator, sets the given parameters and fits it on (X, y)
def fit_with_params(spec, params, X, y):
    model = clone(spec.estimator)
    if params:
        model.set_params(**params)
    return model.fit(X, y)


# Probability of the up-day class. If the model was trained on data with no up days at all,
# zeros are returned rather than raising an error.
def predict_proba_up(model, X):
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(len(X))
