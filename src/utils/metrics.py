from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error

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


def compute_imputation_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute comprehensive imputation metrics (per-feature + overall)."""
    if true_values.shape != predicted_values.shape:
        raise ValueError("True and predicted values must have the same shape")

    n_samples, n_features = true_values.shape

    overall_r2 = r2_score(true_values.flatten(), predicted_values.flatten())
    overall_rmse = float(np.sqrt(mean_squared_error(true_values.flatten(), predicted_values.flatten())))
    overall_corr, overall_corr_p = pearsonr(true_values.flatten(), predicted_values.flatten())

    feature_r2_scores = np.empty(n_features)
    feature_correlations = np.empty(n_features)
    feature_rmse_scores = np.empty(n_features)

    import warnings
    for i in range(n_features):
        true_feat = true_values[:, i]
        pred_feat = predicted_values[:, i]
        feature_r2_scores[i] = r2_score(true_feat, pred_feat)
        if np.std(true_feat) > 0 and np.std(pred_feat) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                feature_correlations[i], _ = pearsonr(true_feat, pred_feat)
        else:
            feature_correlations[i] = np.nan
        feature_rmse_scores[i] = float(np.sqrt(mean_squared_error(true_feat, pred_feat)))

    return {
        "overall_r2": float(overall_r2),
        "overall_rmse": overall_rmse,
        "overall_correlation": float(overall_corr),
        "overall_correlation_pvalue": float(overall_corr_p),
        "mean_feature_r2": float(np.nanmean(feature_r2_scores)),
        "std_feature_r2": float(np.nanstd(feature_r2_scores)),
        "median_feature_r2": float(np.nanmedian(feature_r2_scores)),
        "min_feature_r2": float(np.nanmin(feature_r2_scores)),
        "max_feature_r2": float(np.nanmax(feature_r2_scores)),
        "mean_feature_correlation": float(np.nanmean(feature_correlations)),
        "std_feature_correlation": float(np.nanstd(feature_correlations)),
        "median_feature_correlation": float(np.nanmedian(feature_correlations)),
        "min_feature_correlation": float(np.nanmin(feature_correlations)),
        "max_feature_correlation": float(np.nanmax(feature_correlations)),
        "mean_feature_rmse": float(np.nanmean(feature_rmse_scores)),
        "std_feature_rmse": float(np.nanstd(feature_rmse_scores)),
        "median_feature_rmse": float(np.nanmedian(feature_rmse_scores)),
        "min_feature_rmse": float(np.nanmin(feature_rmse_scores)),
        "max_feature_rmse": float(np.nanmax(feature_rmse_scores)),
        "features_r2_above_0.3": int(np.nansum(feature_r2_scores > 0.3)),
        "features_r2_above_0.5": int(np.nansum(feature_r2_scores > 0.5)),
        "features_r2_above_0.7": int(np.nansum(feature_r2_scores > 0.7)),
        "fraction_r2_above_0.3": float(np.nanmean(feature_r2_scores > 0.3)),
        "fraction_r2_above_0.5": float(np.nanmean(feature_r2_scores > 0.5)),
        "fraction_r2_above_0.7": float(np.nanmean(feature_r2_scores > 0.7)),
        "features_corr_above_0.6": int(np.nansum(feature_correlations > 0.6)),
        "features_corr_above_0.8": int(np.nansum(feature_correlations > 0.8)),
        "fraction_corr_above_0.6": float(np.nanmean(feature_correlations > 0.6)),
        "fraction_corr_above_0.8": float(np.nanmean(feature_correlations > 0.8)),
    }
