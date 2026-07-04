from __future__ import annotations

from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd


def lgb_pearson_eval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    labels = dataset.get_label()
    if labels.std() == 0 or preds.std() == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(labels, preds)[0, 1])
    return "pearson", pearson, True


def train_lgbm_fold_importance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    feature_names: list[str],
    learning_rate: float = 0.01,
    num_leaves: int = 31,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 200,
    min_data_in_leaf: int = 200,
    lambda_l2: float = 10.0,
) -> tuple[lgb.Booster, pd.DataFrame]:
    params = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": lambda_l2,
        "seed": 42,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        feval=lgb_pearson_eval,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(100),
        ],
    )
    importance = model.feature_importance(importance_type="gain")
    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    return model, importance_frame


def collect_fold_top_features(
    importance_frame: pd.DataFrame,
    *,
    fold: int,
    top_k: int = 20,
) -> pd.DataFrame:
    top = importance_frame.head(top_k).copy()
    top["fold"] = fold
    top["rank"] = np.arange(1, len(top) + 1)
    return top


def select_stable_features(
    fold_importance: pd.DataFrame,
    *,
    top_k: int = 20,
    min_folds: int = 3,
) -> list[str]:
    counts: Counter[str] = Counter()
    for _, group in fold_importance.groupby("fold"):
        top_features = group.sort_values("rank").head(top_k)["feature"].tolist()
        counts.update(top_features)

    stable = [feature for feature, count in counts.items() if count >= min_folds]
    stable.sort(key=lambda feature: (-counts[feature], feature))
    return stable


def importance_summary(fold_importance: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_importance.groupby("feature")
        .agg(
            fold_count=("fold", "nunique"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            best_rank=("rank", "min"),
        )
        .reset_index()
        .sort_values(["fold_count", "mean_importance"], ascending=[False, False])
    )
    return summary
