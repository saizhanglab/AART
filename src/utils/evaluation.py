from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .mapping import canonicalize_symbol
from .metrics import build_gene_metrics_table, build_sample_metrics_table, compute_prediction_summary, compute_summary_from_tables


def evaluate_predictions(
    *,
    ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_names: list[str],
    union_lookup: dict[str, bool] | None = None,
    id_column: str = "csid",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    gene_df = build_gene_metrics_table(y_true=y_true, y_pred=y_pred, feature_names=gene_names, union_lookup=union_lookup or {})
    sample_df = build_sample_metrics_table(ids=ids, y_true=y_true, y_pred=y_pred, id_column=id_column)
    summary = compute_summary_from_tables(gene_df=gene_df, sample_df=sample_df)
    summary["n_genes"] = int(len(gene_df))
    summary["n_samples"] = int(len(sample_df))
    summary.update({f"raw_{key}": value for key, value in compute_prediction_summary(y_true, y_pred).items()})
    return gene_df, sample_df, summary


def save_prediction_frame(ids: np.ndarray, predictions: np.ndarray, gene_names: list[str], output_path: str | Path, id_column: str = "csid") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(predictions, columns=gene_names)
    df.insert(0, id_column, ids)
    df.to_csv(output_path, index=False)
    return output_path


def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def join_against_baseline(aart_gene_df: pd.DataFrame, baseline_df: pd.DataFrame, prefix: str = "aart", baseline_name: str = "baseline") -> pd.DataFrame:
    renamed = aart_gene_df.rename(
        columns={
            "pearson_r": f"pearson_r_{prefix}",
            "spearman_r": f"spearman_r_{prefix}",
            "r2": f"r2_{prefix}",
            "rmse": f"rmse_{prefix}",
            "mae": f"mae_{prefix}",
            "n_samples": f"n_samples_{prefix}",
            "best_feature_id": f"best_feature_id_{prefix}",
        }
    )
    merged = renamed.merge(baseline_df, on="protein_symbol", how="inner", suffixes=("", f"_{baseline_name}"))
    merged[f"delta_vs_{baseline_name}_pearson"] = merged[f"pearson_r_{prefix}"] - merged["pearson_r"]
    merged[f"delta_vs_{baseline_name}_spearman"] = merged[f"spearman_r_{prefix}"] - merged["spearman_r"]
    merged[f"delta_vs_{baseline_name}_r2"] = merged[f"r2_{prefix}"] - merged["r2"]
    return merged
