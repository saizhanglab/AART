from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .mapping import canonicalize_symbol


def safe_corr(fn: Any, truth: np.ndarray, pred: np.ndarray) -> float:
    if len(truth) < 2:
        return math.nan
    if np.allclose(np.nanstd(truth), 0.0) or np.allclose(np.nanstd(pred), 0.0):
        return math.nan
    try:
        return float(fn(truth, pred)[0])
    except Exception:
        return math.nan


def gene_corrs(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    yt_centered = yt - yt.mean(axis=0, keepdims=True)
    yp_centered = yp - yp.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.square(yt_centered).sum(axis=0) * np.square(yp_centered).sum(axis=0))
    return np.divide((yt_centered * yp_centered).sum(axis=0), denom, out=np.full(yt.shape[1], np.nan), where=denom > 0)


def sample_corrs(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    yt_centered = yt - yt.mean(axis=1, keepdims=True)
    yp_centered = yp - yp.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.square(yt_centered).sum(axis=1) * np.square(yp_centered).sum(axis=1))
    return np.divide((yt_centered * yp_centered).sum(axis=1), denom, out=np.full(yt.shape[0], np.nan), where=denom > 0)


def median_gene_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmedian(gene_corrs(y_true, y_pred)))


def compute_vector_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[mask]
    pred = pred[mask]
    if len(truth) == 0:
        return {
            "pearson_r": math.nan,
            "spearman_r": math.nan,
            "r2": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "n_samples": 0,
        }
    ss_res = float(np.square(truth - pred).sum())
    ss_tot = float(np.square(truth - truth.mean()).sum())
    r2 = math.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return {
        "pearson_r": safe_corr(pearsonr, truth, pred),
        "spearman_r": safe_corr(spearmanr, truth, pred),
        "r2": r2,
        "rmse": float(np.sqrt(np.square(truth - pred).mean())),
        "mae": float(np.abs(truth - pred).mean()),
        "n_samples": int(len(truth)),
    }


def compute_prediction_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    gene_r = gene_corrs(y_true, y_pred)
    sample_r = sample_corrs(y_true, y_pred)
    ss_res = np.square(y_true - y_pred).sum(axis=0)
    ss_tot = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum(axis=0)
    gene_r2 = 1.0 - np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan, dtype=float), where=ss_tot > 0)
    return {
        "gene_pearson_median": float(np.nanmedian(gene_r)),
        "gene_pearson_mean": float(np.nanmean(gene_r)),
        "gene_spearman_median": float(
            np.nanmedian([safe_corr(spearmanr, y_true[:, idx], y_pred[:, idx]) for idx in range(y_true.shape[1])])
        ),
        "sample_pearson_median": float(np.nanmedian(sample_r)),
        "sample_pearson_mean": float(np.nanmean(sample_r)),
        "gene_r2_median": float(np.nanmedian(gene_r2)),
        "gene_r2_mean": float(np.nanmean(gene_r2)),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def build_gene_metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    union_lookup: dict[str, bool] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    union_lookup = union_lookup or {}
    for feature_index, feature_name in enumerate(feature_names):
        metrics = compute_vector_metrics(y_true[:, feature_index], y_pred[:, feature_index])
        protein_symbol = canonicalize_symbol(feature_name) or feature_name
        rows.append(
            {
                "protein_symbol": protein_symbol,
                "best_feature_id": feature_name,
                **metrics,
                "n_target_features": 1,
                "is_union_p01": bool(union_lookup.get(protein_symbol, False)),
            }
        )
    return pd.DataFrame(rows)


def build_sample_metrics_table(
    ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id_column: str = "csid",
    metric_column: str = "sample_pearson",
) -> pd.DataFrame:
    return pd.DataFrame({id_column: ids, metric_column: sample_corrs(y_true, y_pred)})


def compute_summary_from_tables(gene_df: pd.DataFrame, sample_df: pd.DataFrame, sample_column: str = "sample_pearson") -> dict[str, float]:
    return {
        "gene_pearson_median": float(gene_df["pearson_r"].median()),
        "gene_pearson_mean": float(gene_df["pearson_r"].mean()),
        "gene_spearman_median": float(gene_df["spearman_r"].median()),
        "gene_spearman_mean": float(gene_df["spearman_r"].mean()),
        "gene_r2_median": float(gene_df["r2"].median()),
        "gene_r2_mean": float(gene_df["r2"].mean()),
        "rmse_median": float(gene_df["rmse"].median()),
        "mae_median": float(gene_df["mae"].median()),
        "sample_pearson_median": float(sample_df[sample_column].median()),
        "sample_pearson_mean": float(sample_df[sample_column].mean()),
    }
