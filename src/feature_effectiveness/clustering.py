from __future__ import annotations

import numpy as np
import pandas as pd


def get_anonymous_feature_names(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("X") and column[1:].isdigit()]


def compute_abs_correlation_matrix(
    data: pd.DataFrame,
    feature_names: list[str],
    *,
    sample_rows: int | None = 100_000,
) -> pd.DataFrame:
    frame = data[feature_names]
    if sample_rows is not None and len(frame) > sample_rows:
        frame = frame.iloc[:sample_rows]
    corr = frame.corr(method="pearson").abs().fillna(0.0)
    for idx in corr.index:
        corr.at[idx, idx] = 1.0
    return corr


def build_correlation_clusters(
    corr_matrix: pd.DataFrame,
    *,
    threshold: float = 0.6,
) -> list[list[str]]:
    features = list(corr_matrix.columns)
    unassigned = set(features)
    clusters: list[list[str]] = []

    while unassigned:
        seed = min(unassigned)
        cluster = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            neighbors = [
                feature
                for feature in unassigned
                if feature != current and corr_matrix.at[current, feature] >= threshold
            ]
            for neighbor in neighbors:
                if neighbor not in cluster:
                    cluster.add(neighbor)
                    queue.append(neighbor)
        cluster_list = sorted(cluster)
        clusters.append(cluster_list)
        unassigned -= cluster

    return clusters


def select_cluster_medoid(cluster: list[str], corr_matrix: pd.DataFrame) -> str:
    if len(cluster) == 1:
        return cluster[0]
    sub = corr_matrix.loc[cluster, cluster]
    mean_corr = sub.mean(axis=1)
    return str(mean_corr.idxmax())


def select_medoid_features(
    data: pd.DataFrame,
    feature_names: list[str],
    *,
    threshold: float = 0.6,
    sample_rows: int | None = 100_000,
) -> tuple[list[str], list[dict[str, object]]]:
    corr_matrix = compute_abs_correlation_matrix(
        data,
        feature_names,
        sample_rows=sample_rows,
    )
    clusters = build_correlation_clusters(corr_matrix, threshold=threshold)
    medoids: list[str] = []
    cluster_rows: list[dict[str, object]] = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        medoid = select_cluster_medoid(cluster, corr_matrix)
        medoids.append(medoid)
        for feature in cluster:
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "feature": feature,
                    "medoid": medoid,
                    "cluster_size": len(cluster),
                    "is_medoid": feature == medoid,
                }
            )
    return medoids, cluster_rows


def compute_feature_target_correlation(
    data: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    *,
    sample_rows: int | None = None,
) -> pd.Series:
    frame = data
    if sample_rows is not None and len(frame) > sample_rows:
        frame = frame.iloc[:sample_rows]
    y = frame[target_col].astype(np.float64)
    corrs = {}
    for feature in feature_names:
        x = frame[feature].astype(np.float64)
        if x.std() == 0 or y.std() == 0:
            corrs[feature] = 0.0
        else:
            corrs[feature] = float(x.corr(y, method="pearson"))
    return pd.Series(corrs)


def filter_low_signal_features(
    target_corr: pd.Series,
    *,
    threshold: float = 1e-4,
) -> tuple[list[str], list[str]]:
    kept = [feature for feature, corr in target_corr.items() if abs(corr) > threshold]
    dropped = [feature for feature, corr in target_corr.items() if abs(corr) <= threshold]
    return kept, dropped
