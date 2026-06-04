#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aart.anchors import AnchorBundle, apply_gate, compute_residual_prior, fit_closed_form_gate, fit_residual_bundle
from src.models.aart.data import load_paired_tables, split_external_train_for_validation
from src.models.aart.evaluation import evaluate_predictions, save_prediction_frame, write_json
from src.models.aart.mapping import build_candidate_table
from src.models.aart.metrics import median_gene_pearson
from src.models.aart.plotting import save_gene_histogram, save_scatter
from src.models.aart.preprocessing import dataframe_ids, dataframe_targets, fit_preprocessing, inverse_transform_targets, transform_olink, transform_soma


METHOD_NAMES = ["direct_anchor", "hybrid_ridge", "aart_closed_form_gate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fair-holdout CPU baselines for SomaScan -> Olink")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--reuse-sweep", action="store_true", help="Reuse existing sweep_results.csv and best_configs.json")
    return parser.parse_args()


def fit_and_predict(config_row: dict[str, float], x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, gene_names: list[str], candidates: pd.DataFrame, random_state: int) -> dict[str, object]:
    anchor_bundle = AnchorBundle.fit(
        x_train=x_train,
        y_train=y_train,
        gene_names=gene_names,
        candidates=candidates,
        alpha_single=float(config_row["anchor_alpha_single"]),
        alpha_multi=float(config_row["anchor_alpha_multi"]),
    )
    anchor_train = anchor_bundle.predict(x_train)
    anchor_val = anchor_bundle.predict(x_val)
    anchor_test = anchor_bundle.predict(x_test)

    residual_bundle = fit_residual_bundle(
        x_train=x_train,
        residual_train=y_train - anchor_train,
        n_components=int(config_row["n_components"]),
        alpha=float(config_row["residual_alpha"]),
        random_state=random_state,
    )
    residual_val = residual_bundle.predict(x_val)
    residual_test = residual_bundle.predict(x_test)
    hybrid_val = anchor_val + residual_val
    hybrid_test = anchor_test + residual_test
    residual_prior = compute_residual_prior(anchor_bundle)
    gate = fit_closed_form_gate(
        residual_val_pred=residual_val,
        anchor_val=anchor_val,
        y_val=y_val,
        residual_prior=residual_prior,
        lam=float(config_row["gate_lambda"]),
    )
    aart_val = apply_gate(anchor_val, residual_val, gate)
    aart_test = apply_gate(anchor_test, residual_test, gate)
    return {
        "anchor_bundle": anchor_bundle,
        "gate": gate,
        "direct_val": anchor_val,
        "hybrid_val": hybrid_val,
        "aart_val": aart_val,
        "direct_test": anchor_test,
        "hybrid_test": hybrid_test,
        "aart_test": aart_test,
    }


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    split_config = config["split"]
    outdir = PROJECT_ROOT / "outputs" / "fair_holdout" / "baselines"
    figures_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    predictions_dir = outdir / "predictions"
    for path in [outdir, figures_dir, tables_dir, predictions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    train_olink, train_soma = load_paired_tables(paths["train_olink"], paths["train_soma"], id_column=split_config["id_column"])
    test_olink, test_soma = load_paired_tables(paths["test_olink"], paths["test_soma"], id_column=split_config["id_column"])
    annotation = pd.read_csv(paths["annotation"])

    split_frames = split_external_train_for_validation(
        olink_train=train_olink,
        soma_train=train_soma,
        val_size=float(split_config["val_size"]),
        random_state=int(split_config["random_state"]),
        id_column=split_config["id_column"],
    )
    preprocessing, x_train, y_train = fit_preprocessing(split_frames.olink_train, split_frames.soma_train, id_column=split_config["id_column"])
    x_val = transform_soma(split_frames.soma_val, preprocessing)
    y_val = transform_olink(split_frames.olink_val, preprocessing)
    x_test = transform_soma(test_soma, preprocessing)
    y_test_raw = dataframe_targets(test_olink, id_column=split_config["id_column"])
    ids_test = dataframe_ids(test_olink, id_column=split_config["id_column"])

    candidates = build_candidate_table(preprocessing.gene_names, preprocessing.soma_feature_names, annotation)
    candidates.to_csv(tables_dir / "candidate_mapping.csv", index=False)

    grid = config["baseline_grid"]
    sweep_results_path = tables_dir / "sweep_results.csv"
    best_configs_path = outdir / "best_configs.json"
    best_by_method: dict[str, dict[str, object]] = {}
    if args.reuse_sweep and sweep_results_path.exists() and best_configs_path.exists():
        sweep_df = pd.read_csv(sweep_results_path)
        best_configs = json.loads(best_configs_path.read_text())
        score_cols = {
            "direct_anchor": "val_direct_anchor",
            "hybrid_ridge": "val_hybrid_ridge",
            "aart_closed_form_gate": "val_aart_closed_form_gate",
        }
        for method_name, config_row in best_configs.items():
            mask = pd.Series(True, index=sweep_df.index)
            for key, value in config_row.items():
                if value is None or key not in sweep_df.columns:
                    continue
                mask &= sweep_df[key] == value
            score = float(sweep_df.loc[mask, score_cols[method_name]].max()) if mask.any() else float("nan")
            best_by_method[method_name] = {"score": score, "config": config_row}
    else:
        sweep_rows: list[dict[str, float]] = []
        best_by_method = {}
        anchor_cache: dict[tuple[float, float], dict[str, object]] = {}
        residual_cache: dict[tuple[float, float, float, int], dict[str, object]] = {}
        for anchor_alpha_single, anchor_alpha_multi in itertools.product(grid["anchor_alpha_single"], grid["anchor_alpha_multi"]):
            anchor_key = (float(anchor_alpha_single), float(anchor_alpha_multi))
            anchor_bundle = AnchorBundle.fit(
                x_train=x_train,
                y_train=y_train,
                gene_names=preprocessing.gene_names,
                candidates=candidates,
                alpha_single=anchor_key[0],
                alpha_multi=anchor_key[1],
            )
            anchor_train = anchor_bundle.predict(x_train)
            anchor_val = anchor_bundle.predict(x_val)
            anchor_cache[anchor_key] = {
                "anchor_bundle": anchor_bundle,
                "anchor_train": anchor_train,
                "anchor_val": anchor_val,
                "direct_score": median_gene_pearson(y_val, anchor_val),
            }
            direct_config = {
                "anchor_alpha_single": anchor_key[0],
                "anchor_alpha_multi": anchor_key[1],
                "residual_alpha": None,
                "n_components": None,
                "gate_lambda": None,
            }
            current_best = best_by_method.get("direct_anchor")
            if current_best is None or anchor_cache[anchor_key]["direct_score"] > current_best["score"]:
                best_by_method["direct_anchor"] = {"score": anchor_cache[anchor_key]["direct_score"], "config": direct_config}

        for anchor_alpha_single, anchor_alpha_multi, residual_alpha, n_components, gate_lambda in itertools.product(
            grid["anchor_alpha_single"],
            grid["anchor_alpha_multi"],
            grid["residual_alpha"],
            grid["n_components"],
            grid["gate_lambda"],
        ):
            anchor_key = (float(anchor_alpha_single), float(anchor_alpha_multi))
            cache_entry = anchor_cache[anchor_key]
            config_row = {
                "anchor_alpha_single": anchor_key[0],
                "anchor_alpha_multi": anchor_key[1],
                "residual_alpha": float(residual_alpha),
                "n_components": int(n_components),
                "gate_lambda": float(gate_lambda),
            }
            residual_key = (anchor_key[0], anchor_key[1], float(residual_alpha), int(n_components))
            if residual_key not in residual_cache:
                residual_bundle = fit_residual_bundle(
                    x_train=x_train,
                    residual_train=y_train - cache_entry["anchor_train"],
                    n_components=int(n_components),
                    alpha=float(residual_alpha),
                    random_state=int(split_config["random_state"]),
                )
                residual_val = residual_bundle.predict(x_val)
                residual_cache[residual_key] = {
                    "residual_bundle": residual_bundle,
                    "residual_val": residual_val,
                    "hybrid_val": cache_entry["anchor_val"] + residual_val,
                }
            residual_entry = residual_cache[residual_key]
            residual_prior = compute_residual_prior(cache_entry["anchor_bundle"])
            gate = fit_closed_form_gate(
                residual_val_pred=residual_entry["residual_val"],
                anchor_val=cache_entry["anchor_val"],
                y_val=y_val,
                residual_prior=residual_prior,
                lam=float(gate_lambda),
            )
            aart_val = apply_gate(cache_entry["anchor_val"], residual_entry["residual_val"], gate)
            scores = {
                "direct_anchor": float(cache_entry["direct_score"]),
                "hybrid_ridge": median_gene_pearson(y_val, residual_entry["hybrid_val"]),
                "aart_closed_form_gate": median_gene_pearson(y_val, aart_val),
            }
            sweep_rows.append({**config_row, **{f"val_{key}": value for key, value in scores.items()}})
            for method_name in ["hybrid_ridge", "aart_closed_form_gate"]:
                score = scores[method_name]
                current_best = best_by_method.get(method_name)
                if current_best is None or score > current_best["score"]:
                    best_by_method[method_name] = {"score": score, "config": config_row}

        sweep_df = pd.DataFrame(sweep_rows).sort_values("val_aart_closed_form_gate", ascending=False)
        sweep_df.to_csv(sweep_results_path, index=False)

    per_method_gene_tables: dict[str, pd.DataFrame] = {}
    per_method_sample_tables: dict[str, pd.DataFrame] = {}
    method_summary_rows: list[dict[str, object]] = []
    baseline_predictions: dict[str, np.ndarray] = {}
    gated_metadata: pd.DataFrame | None = None

    for method_name in METHOD_NAMES:
        chosen_config = best_by_method[method_name]["config"]
        if method_name == "direct_anchor":
            anchor_bundle = AnchorBundle.fit(
                x_train=x_train,
                y_train=y_train,
                gene_names=preprocessing.gene_names,
                candidates=candidates,
                alpha_single=float(chosen_config["anchor_alpha_single"]),
                alpha_multi=float(chosen_config["anchor_alpha_multi"]),
            )
            pred_std = anchor_bundle.predict(x_test)
            if gated_metadata is None:
                metadata = anchor_bundle.metadata_frame()
                metadata["gate_residual_weight"] = np.nan
                metadata["gate_anchor_weight"] = np.nan
                gated_metadata = metadata
        else:
            fitted = fit_and_predict(
                config_row=chosen_config,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_train=y_train,
                y_val=y_val,
                gene_names=preprocessing.gene_names,
                candidates=candidates,
                random_state=int(split_config["random_state"]),
            )
            pred_std = fitted["hybrid_test"] if method_name == "hybrid_ridge" else fitted["aart_test"]
        pred_raw = inverse_transform_targets(pred_std, preprocessing)
        baseline_predictions[method_name] = pred_raw
        save_prediction_frame(
            ids=ids_test,
            predictions=pred_raw,
            gene_names=preprocessing.gene_names,
            output_path=predictions_dir / f"{method_name}_test_predictions.csv",
            id_column=split_config["id_column"],
        )
        gene_df, sample_df, summary = evaluate_predictions(
            ids=ids_test,
            y_true=y_test_raw,
            y_pred=pred_raw,
            gene_names=preprocessing.gene_names,
            id_column=split_config["id_column"],
        )
        per_method_gene_tables[method_name] = gene_df
        per_method_sample_tables[method_name] = sample_df.rename(columns={"sample_pearson": f"sample_pearson_{method_name}"})
        method_summary_rows.append({"method": method_name, **chosen_config, "validation_gene_pearson_median": best_by_method[method_name]["score"], **summary})

        if method_name == "aart_closed_form_gate":
            metadata = fitted["anchor_bundle"].metadata_frame()
            metadata["gate_residual_weight"] = fitted["gate"]
            metadata["gate_anchor_weight"] = 1.0 - fitted["gate"]
            gated_metadata = metadata

    method_summary_df = pd.DataFrame(method_summary_rows).sort_values("gene_pearson_median", ascending=False)
    method_summary_df.to_csv(outdir / "method_summary.csv", index=False)

    per_gene_wide: pd.DataFrame | None = None
    for method_name, gene_df in per_method_gene_tables.items():
        renamed = gene_df.rename(
            columns={
                "best_feature_id": f"best_feature_id_{method_name}",
                "pearson_r": f"pearson_r_{method_name}",
                "spearman_r": f"spearman_r_{method_name}",
                "r2": f"r2_{method_name}",
                "rmse": f"rmse_{method_name}",
                "mae": f"mae_{method_name}",
                "n_samples": f"n_samples_{method_name}",
                "n_target_features": f"n_target_features_{method_name}",
            }
        )
        keep_columns = ["protein_symbol", "is_union_p01"] + [column for column in renamed.columns if column not in {"protein_symbol", "is_union_p01"}]
        renamed = renamed[keep_columns]
        if per_gene_wide is None:
            per_gene_wide = renamed.copy()
        else:
            per_gene_wide = per_gene_wide.merge(renamed.drop(columns=["is_union_p01"]), on="protein_symbol", how="outer")
    if per_gene_wide is None:
        raise RuntimeError("No per-gene baseline outputs were produced")
    if gated_metadata is not None:
        per_gene_wide = per_gene_wide.merge(
            gated_metadata[
                [
                    "gene",
                    "protein_symbol",
                    "match_type",
                    "n_candidates_used",
                    "anchor_features",
                    "train_anchor_corr",
                    "has_anchor",
                    "gate_residual_weight",
                    "gate_anchor_weight",
                ]
            ],
            on="protein_symbol",
            how="left",
        )
    per_gene_wide.to_csv(outdir / "per_gene_metrics.csv", index=False)

    sample_wide = pd.DataFrame({split_config["id_column"]: ids_test})
    for sample_df in per_method_sample_tables.values():
        sample_wide = sample_wide.merge(sample_df, on=split_config["id_column"], how="left")
    sample_wide.to_csv(outdir / "sample_metrics.csv", index=False)

    metrics_long_rows = []
    for method_name, gene_df in per_method_gene_tables.items():
        frame = gene_df[["protein_symbol", "pearson_r"]].copy()
        frame["method"] = method_name
        metrics_long_rows.append(frame)
    metrics_long = pd.concat(metrics_long_rows, ignore_index=True)
    metrics_long.to_csv(outdir / "metrics_long.csv", index=False)

    save_gene_histogram(metrics_long, figures_dir / "gene_corr_histograms.png")
    scatter_frame = per_gene_wide.dropna(subset=["pearson_r_direct_anchor", "pearson_r_aart_closed_form_gate"])
    save_scatter(
        frame=scatter_frame,
        x_col="pearson_r_direct_anchor",
        y_col="pearson_r_aart_closed_form_gate",
        output_path=figures_dir / "direct_vs_aart_scatter.png",
        title="Direct anchor vs closed-form AART",
        xlabel="Direct anchor Pearson r",
        ylabel="AART closed-form Pearson r",
    )
    scatter_frame = per_gene_wide.dropna(subset=["pearson_r_hybrid_ridge", "pearson_r_aart_closed_form_gate"])
    save_scatter(
        frame=scatter_frame,
        x_col="pearson_r_hybrid_ridge",
        y_col="pearson_r_aart_closed_form_gate",
        output_path=figures_dir / "hybrid_vs_aart_scatter.png",
        title="Hybrid ridge vs closed-form AART",
        xlabel="Hybrid ridge Pearson r",
        ylabel="AART closed-form Pearson r",
    )

    best_configs = {method_name: payload["config"] for method_name, payload in best_by_method.items()}
    write_json(best_configs, outdir / "best_configs.json")
    write_json({"baseline_summary": method_summary_rows}, outdir / "summary.json")

    lines = ["# CPU baselines on the fair holdout", ""]
    for _, row in method_summary_df.iterrows():
        lines.append(
            f"- `{row['method']}`: test median gene Pearson `{row['gene_pearson_median']:.4f}`, "
            f"validation median gene Pearson `{row['validation_gene_pearson_median']:.4f}`"
        )
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(method_summary_df[["method", "gene_pearson_median", "sample_pearson_median", "validation_gene_pearson_median"]].to_string(index=False))
    print(f"Saved CPU baseline outputs to {outdir}")


if __name__ == "__main__":
    main()
