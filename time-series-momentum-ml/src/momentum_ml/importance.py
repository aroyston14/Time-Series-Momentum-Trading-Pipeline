# Extracting feature importance from the fitted models

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# Logistic regression coefficients in both standardised and raw units.
# coef_per_sd is the change in log-odds for a one standard deviation move in the feature, so it can
# be compared directly across features. coef_raw_units divides that by the scaler's standard
# deviation, giving the change in log-odds per unit of the unscaled feature.
def logistic_coefficients(model, features):
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["clf"]
    coefficients = classifier.coef_.ravel()
    coefficient_table = pd.DataFrame(
        {
            "feature": features,
            "coef_per_sd": coefficients,
            "abs_coef_per_sd": np.abs(coefficients),
            "coef_raw_units": coefficients / scaler.scale_,
            "feature_sd": scaler.scale_,
        }
    )
    return coefficient_table.sort_values("abs_coef_per_sd", ascending=False, ignore_index=True)


# Impurity importance for a random forest, or gain and split counts for XGBoost.
# Returns None if the model has neither.
def tree_importance(model, features):

    # XGBoost models have a booster which reports gain and the number of splits per feature
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        split_counts = booster.get_score(importance_type="weight")
        importance_table = pd.DataFrame(
            {
                "feature": features,
                "gain": [gain_scores.get(feature, 0.0) for feature in features],
                "split_count": [split_counts.get(feature, 0.0) for feature in features],
            }
        )
        total_gain = importance_table["gain"].sum()
        importance_table["importance"] = (
            importance_table["gain"] / total_gain if total_gain > 0 else 0.0
        )
        return importance_table.sort_values("importance", ascending=False, ignore_index=True)

    if hasattr(model, "feature_importances_"):
        importance_table = pd.DataFrame(
            {"feature": features, "importance": model.feature_importances_}
        )
        return importance_table.sort_values("importance", ascending=False, ignore_index=True)
    return None


# Permutation importance on held out data, i.e. how much the score drops when each feature is
# shuffled. If max_rows is set, the most recent max_rows rows are used (a contiguous block rather
# than a random sample).
def permutation_table(model, X, y, scoring, n_repeats, seed, max_rows=None):
    if max_rows and len(X) > max_rows:
        X, y = X.iloc[-max_rows:], y.iloc[-max_rows:]
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    importance_table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance_table.sort_values("importance_mean", ascending=False, ignore_index=True)


# Works out every importance table for the given models, which should have been fitted on the
# training set. The keys look like coef_logreg_l2, impurity_random_forest, gain_xgboost and
# permutation_<model name>.
def all_importances(models, X_eval, y_eval, config):
    features = list(X_eval.columns)
    permutation_config = config["evaluation"].get("permutation_importance", {})
    seed = int(config["project"]["seed"])

    importances = {}
    for name, model in models.items():
        if name == "majority":
            continue

        # Logistic regressions are the pipelines with a scaler, everything else is a tree model
        if isinstance(model, Pipeline) and "scaler" in model.named_steps:
            importances[f"coef_{name}"] = logistic_coefficients(model, features)
        else:
            tree_table = tree_importance(model, features)
            if tree_table is not None:
                kind = "gain" if hasattr(model, "get_booster") else "impurity"
                importances[f"{kind}_{name}"] = tree_table
            else:
                logger.info("%s exposes no native importance; permutation importance only", name)

        if permutation_config.get("enabled", True):
            importances[f"permutation_{name}"] = permutation_table(
                model,
                X_eval,
                y_eval,
                permutation_config.get("scoring", "roc_auc"),
                int(permutation_config.get("n_repeats", 10)),
                seed,
                permutation_config.get("max_rows"),
            )
    return importances
