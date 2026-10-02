# Command line entry point that tunes, selects and fits the models, then saves the predictions
# and the model metadata.
#
# The protocol is:
#   1. Split the data chronologically into train, validation and test (purged at the boundaries).
#   2. Tune each model on the training split with date-grouped TimeSeriesSplit CV.
#   3. Use the tuned models (fitted on train) to predict the validation split. The final model is
#      the non-baseline model with the best validation tuning.selection_metric. The test split
#      plays no part in steps 2 and 3.
#   4. Refit every model on train + validation with its chosen hyperparameters and predict the
#      test split once.
#   5. Optionally run an expanding window walk-forward evaluation, which refits each model (with
#      the hyperparameters fixed from step 2) on all the data before each block and predicts it.
#
# Example:
#     python -m momentum_ml.train --config configs/config.yaml

# Standard library modules for the command line arguments, saving metadata, logging and
# recording the python version
import argparse
import json
import logging
import platform

# joblib is used to save the fitted models to disk
import joblib
import numpy as np
import pandas as pd
import sklearn

from momentum_ml.build_features import load_panel
from momentum_ml.config import load_config, resolve_path, set_seeds, setup_logging
from momentum_ml.features import FWD_RET_COL, TARGET_COL, feature_columns
from momentum_ml.metrics import better, classification_metrics
from momentum_ml.models import (
    build_model_specs,
    fit_with_params,
    predict_proba_up,
    tune_model,
    xgboost_available,
)
from momentum_ml.splits import chronological_split, purge_days, walk_forward_folds

logger = logging.getLogger("momentum_ml.train")


# Makes a model name safe to use in a file name, e.g. "xgboost[sklearn-hgb-fallback]"
def _safe(name):
    return name.replace("[", "_").replace("]", "").replace("/", "_")


# Puts one model's predicted probabilities for a split into a tidy DataFrame
def _predictions_frame(df, model_name, prob_up, split_name):
    return pd.DataFrame(
        {
            "date": df["date"].to_numpy(),
            "ticker": df["ticker"].to_numpy(),
            "split": split_name,
            "model": model_name,
            "prob_up": prob_up,
            TARGET_COL: df[TARGET_COL].to_numpy(),
            FWD_RET_COL: df[FWD_RET_COL].to_numpy(),
        }
    )


# Creates the output folders from the config (if they do not already exist) and returns them
def output_dirs(config):
    outputs_config = config["outputs"]
    dirs = {
        "models": resolve_path(config, outputs_config["models_dir"]),
        "tables": resolve_path(config, outputs_config["tables_dir"]),
        "figures": resolve_path(config, outputs_config["figures_dir"]),
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    (dirs["models"] / "train_fit").mkdir(exist_ok=True)
    (dirs["models"] / "final_fit").mkdir(exist_ok=True)
    return dirs


# Runs the training protocol described at the top of this file, saves everything it produces
# and returns the metadata dict
def run(config):
    seed = int(config["project"]["seed"])
    set_seeds(seed)
    dirs = output_dirs(config)
    panel = load_panel(config)
    features = feature_columns(panel)

    splits, split_infos = chronological_split(panel, config)
    pd.DataFrame([info.as_dict() for info in split_infos]).to_csv(
        dirs["tables"] / "splits.csv", index=False
    )
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    # Steps 2 and 3: tune every model on the training data and score it on the validation data
    specs = build_model_specs(config)
    tuned_models = {}
    validation_predictions = []
    validation_scores = []
    for name, spec in specs.items():
        logger.info("Tuning %s on training data (%d rows)", name, len(train))
        tuned = tune_model(spec, train[features], train[TARGET_COL], train["date"], config)
        tuned_models[name] = tuned
        if tuned.cv_results is not None:
            tuned.cv_results.to_csv(dirs["tables"] / f"tuning_cv_{_safe(name)}.csv", index=False)
        joblib.dump(tuned.estimator, dirs["models"] / "train_fit" / f"{_safe(name)}.joblib")

        prob_up = predict_proba_up(tuned.estimator, validation[features])
        validation_predictions.append(_predictions_frame(validation, name, prob_up, "validation"))
        validation_scores.append(
            {"model": name, **classification_metrics(validation[TARGET_COL].to_numpy(), prob_up)}
        )

    # Picking the best non-baseline model on the validation selection metric
    selection_metric = config["tuning"].get("selection_metric", "log_loss")
    selection_table = pd.DataFrame(validation_scores)
    candidates = selection_table[selection_table["model"] != "majority"]
    best_name, best_score = None, float("nan")
    for _, row in candidates.iterrows():
        if best_name is None or better(selection_metric, row[selection_metric], best_score):
            best_name, best_score = row["model"], row[selection_metric]

    # Recording whether the chosen model actually beat the majority class baseline
    majority_score = selection_table.loc[selection_table["model"] == "majority", selection_metric]
    beats_majority = bool(
        len(majority_score) and better(selection_metric, best_score, float(majority_score.iloc[0]))
    )
    selection_table["selected"] = selection_table["model"] == best_name
    selection_table.to_csv(dirs["tables"] / "model_selection_validation.csv", index=False)
    logger.info(
        "Selected %s by validation %s = %.5f (beats majority baseline: %s)",
        best_name,
        selection_metric,
        best_score,
        beats_majority,
    )

    # Step 4: refit on train + validation and make a single pass over the test set
    if config["evaluation"].get("refit_on_train_val", True):
        development = pd.concat([train, validation])
    else:
        development = train
    test_predictions = []
    for name, tuned in tuned_models.items():
        final_model = fit_with_params(
            tuned.spec, tuned.best_params, development[features], development[TARGET_COL]
        )
        joblib.dump(final_model, dirs["models"] / "final_fit" / f"{_safe(name)}.joblib")
        test_predictions.append(
            _predictions_frame(test, name, predict_proba_up(final_model, test[features]), "test")
        )

    # Step 5: walk-forward evaluation. This is only used for evaluation, the hyperparameters stay
    # fixed from the tuning in step 2.
    walk_forward_predictions = []
    walk_forward_fold_rows = []
    walk_forward_config = config.get("walk_forward", {})
    if walk_forward_config.get("enabled", True):
        if walk_forward_config.get("scope", "full") == "full":
            scope_panel = panel
        else:
            scope_panel = pd.concat([train, validation])

        for fold in walk_forward_folds(
            scope_panel["date"],
            int(walk_forward_config["min_train_days"]),
            int(walk_forward_config["test_days"]),
            purge_days(config),
        ):
            fold_train = scope_panel[scope_panel["date"].isin(fold.train_dates)]
            fold_test = scope_panel[scope_panel["date"].isin(fold.test_dates)]
            walk_forward_fold_rows.append(
                {
                    "fold": fold.fold,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "train_rows": len(fold_train),
                    "test_rows": len(fold_test),
                }
            )
            logger.info(
                "Walk-forward fold %d: train -> %s, test %s -> %s",
                fold.fold,
                fold.train_end,
                fold.test_start,
                fold.test_end,
            )
            for name, tuned in tuned_models.items():
                fold_model = fit_with_params(
                    tuned.spec, tuned.best_params, fold_train[features], fold_train[TARGET_COL]
                )
                prob_up = predict_proba_up(fold_model, fold_test[features])
                fold_predictions = _predictions_frame(fold_test, name, prob_up, "walk_forward")
                fold_predictions["fold"] = fold.fold
                walk_forward_predictions.append(fold_predictions)

        pd.DataFrame(walk_forward_fold_rows).to_csv(
            dirs["tables"] / "walk_forward_folds.csv", index=False
        )

    # Saving all of the out-of-sample predictions
    pd.concat(validation_predictions).to_csv(
        dirs["tables"] / "predictions_validation.csv", index=False
    )
    pd.concat(test_predictions).to_csv(dirs["tables"] / "predictions_test.csv", index=False)
    if walk_forward_predictions:
        pd.concat(walk_forward_predictions).to_csv(
            dirs["tables"] / "predictions_walk_forward.csv", index=False
        )

    # The metadata records everything needed to understand and reproduce the run
    model_info = {}
    for name, tuned in tuned_models.items():
        base_params = {}
        for key, value in tuned.spec.estimator.get_params(deep=False).items():
            if key != "steps":
                base_params[key] = repr(value)
        model_info[name] = {
            "family": tuned.spec.family,
            "estimator": type(tuned.spec.estimator).__name__,
            "base_params": base_params,
            "grid": tuned.spec.grid,
            "best_params": tuned.best_params,
            "cv_score": tuned.cv_score,
            "cv_scoring": config["tuning"].get("scoring"),
            "note": tuned.spec.note,
        }

    metadata = {
        "seed": seed,
        "config_path": config.get("_config_path"),
        "data_source": config["data"]["source"],
        "feature_columns": features,
        "splits": [info.as_dict() for info in split_infos],
        "purge_days": purge_days(config),
        "selection_metric": selection_metric,
        "selected_model": best_name,
        "selected_validation_score": best_score,
        "selected_beats_majority_on_validation": beats_majority,
        "refit_on_train_val": bool(config["evaluation"].get("refit_on_train_val", True)),
        "walk_forward_folds": walk_forward_fold_rows,
        "models": model_info,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost_available": xgboost_available(),
        },
    }
    if xgboost_available():
        import xgboost

        metadata["versions"]["xgboost"] = xgboost.__version__

    (dirs["models"] / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved model metadata to %s", dirs["models"] / "model_metadata.json")
    return metadata


# Entry point for python -m momentum_ml.train
def main(argv=None):
    parser = argparse.ArgumentParser(description="Tune, select and fit models.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config["project"].get("log_level", "INFO"))
    run(config)


if __name__ == "__main__":
    main()
