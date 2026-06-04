#!/usr/bin/env python3
"""
Apply classical AART (SomaScan -> Olink) to GNPC SomaScan data.

Pipeline:
  1. Identify the probe intersection: CKB training probes ∩ GNPC QC'd probes
  2. Retrain AART on CKB using only the intersection probes
  3. Apply to GNPC SomaScan (no patching of absent probes)

AART components: AnchorBundle (per-protein Ridge) + ResidualBundle (PCA-256 + Ridge)
                 + closed-form reliability gate.

Input:  data/GNPC/somascan_qc.parquet   (from data/GNPC/scripts/qc_somascan.py)
Output: data/GNPC_UKBB/gnpc_olink_imputed.parquet
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aart.anchors import (
    AnchorBundle,
    apply_gate,
    compute_residual_prior,
    fit_closed_form_gate,
    fit_residual_bundle,
)
from src.models.aart.mapping import build_candidate_table

DATA_DIR = PROJECT_ROOT / "data"
GNPC_DIR = DATA_DIR / "GNPC"
OUT_DIR = DATA_DIR / "GNPC_UKBB"
ARTIFACTS_PATH = OUT_DIR / "aart_classical_artifacts.pkl"

# Hyperparameters matching run_5fold_cpu.py
ANCHOR_ALPHA_SINGLE = 1.0
ANCHOR_ALPHA_MULTI = 0.1
RESIDUAL_N_COMPONENTS = 256
RESIDUAL_ALPHA = 300.0
GATE_LAMBDA = 1.0
VAL_FRACTION = 0.10


def parse_gnpc_soma_col(col: str) -> str:
    """'seq_10000_28|P43320|CRYBB2' -> 'seq.10000.28'"""
    seqid = col.split("|")[0]
    return seqid.replace("_", ".", 2)


def get_gnpc_available_probes(gnpc_soma_path: Path) -> set[str]:
    """Return the set of probe names (in training format) present in the QC'd parquet."""
    cols = pd.read_parquet(gnpc_soma_path).columns.tolist()
    return {parse_gnpc_soma_col(c) for c in cols}


def load_training_data(
    probe_subset: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], StandardScaler, StandardScaler]:
    """
    Load CKB training data restricted to probe_subset.
    Returns (x_train, y_train, soma_names, gene_names, x_scaler, y_scaler).
    """
    olink = pd.read_csv(DATA_DIR / "olink_overlap_train.csv")
    soma  = pd.read_csv(DATA_DIR / "somascan_overlap_train.csv")

    assert len(olink) == len(soma)
    assert (olink["csid"].values == soma["csid"].values).all()

    all_soma_names = [c for c in soma.columns if c != "csid"]
    gene_names = [c for c in olink.columns if c != "csid"]

    # Restrict to the intersection probes (preserving CKB ordering)
    probe_set = set(probe_subset)
    soma_names = [p for p in all_soma_names if p in probe_set]

    soma_vals = soma[soma_names].values.astype(np.float32)
    olink_vals = olink[gene_names].values.astype(np.float32)

    if np.any(soma_vals <= 0):
        soma_vals = soma_vals + 1.0
    soma_log = np.log(soma_vals)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(soma_log).astype(np.float32)
    y_train = y_scaler.fit_transform(olink_vals).astype(np.float32)

    print(f"Training data: {x_train.shape[0]} samples, "
          f"{x_train.shape[1]} SomaScan probes, {y_train.shape[1]} Olink proteins")
    return x_train, y_train, soma_names, gene_names, x_scaler, y_scaler


def train_classical_aart(
    x_train: np.ndarray,
    y_train: np.ndarray,
    soma_names: list[str],
    gene_names: list[str],
) -> tuple[AnchorBundle, object, np.ndarray]:
    """Fit AnchorBundle + ResidualBundle + closed-form gate."""
    annotation = pd.read_csv(DATA_DIR / "somascan_annotation_cache.csv")
    candidates = build_candidate_table(
        gene_names=gene_names,
        soma_feature_names=soma_names,
        annotation_df=annotation,
    )

    print("Fitting AnchorBundle...")
    anchor_bundle = AnchorBundle.fit(
        x_train=x_train,
        y_train=y_train,
        gene_names=gene_names,
        candidates=candidates,
        alpha_single=ANCHOR_ALPHA_SINGLE,
        alpha_multi=ANCHOR_ALPHA_MULTI,
    )

    anchor_train = anchor_bundle.predict(x_train)
    residual_train = y_train - anchor_train

    print("Fitting ResidualBundle (PCA + Ridge)...")
    residual_bundle = fit_residual_bundle(
        x_train=x_train,
        residual_train=residual_train,
        n_components=RESIDUAL_N_COMPONENTS,
        alpha=RESIDUAL_ALPHA,
        random_state=42,
    )

    val_n = max(1, int(len(x_train) * VAL_FRACTION))
    r_val_pred = residual_bundle.predict(x_train[-val_n:])
    a_val      = anchor_train[-val_n:]
    y_val      = y_train[-val_n:]
    gate_prior = compute_residual_prior(anchor_bundle)

    print(f"Fitting closed-form gate on {val_n} validation samples...")
    gate_weights = fit_closed_form_gate(
        residual_val_pred=r_val_pred,
        anchor_val=a_val,
        y_val=y_val,
        residual_prior=gate_prior,
        lam=GATE_LAMBDA,
    )
    return anchor_bundle, residual_bundle, gate_weights


def load_gnpc_soma_aligned(
    gnpc_soma_path: Path,
    training_soma_names: list[str],
    x_scaler: StandardScaler,
) -> tuple[np.ndarray, list[str]]:
    """
    Load QC'd GNPC SomaScan, select the intersection probes (exact match to
    training_soma_names), median-impute IQR-masked NaN, apply CKB scaler.
    No patching of absent probes — training_soma_names already excludes them.
    """
    df = pd.read_parquet(gnpc_soma_path)   # already ln-transformed
    sample_ids = df.index.tolist()

    col_map = {c: parse_gnpc_soma_col(c) for c in df.columns}
    df = df.rename(columns=col_map)

    # Select exactly the intersection probes in training order
    df = df[training_soma_names]
    print(f"  GNPC: {len(sample_ids)} samples × {len(training_soma_names)} intersection probes")

    values = df.values.astype(np.float32)
    for j in range(values.shape[1]):
        mask = np.isnan(values[:, j])
        if mask.any():
            values[mask, j] = np.nanmedian(values[:, j])

    x_gnpc = x_scaler.transform(values).astype(np.float32)
    return x_gnpc, sample_ids


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "gnpc_olink_imputed.parquet"

    gnpc_soma_path = GNPC_DIR / "somascan_qc.parquet"

    # Determine the intersection: CKB training probes ∩ GNPC QC'd probes
    gnpc_probe_set = get_gnpc_available_probes(gnpc_soma_path)

    # Load cached artifacts only if they were trained on the same probe set
    trained = False
    if ARTIFACTS_PATH.exists():
        with ARTIFACTS_PATH.open("rb") as fh:
            saved = pickle.load(fh)
        if set(saved["soma_names"]) == gnpc_probe_set:
            print(f"Loading cached AART artifacts ({len(saved['soma_names'])} probes)")
            anchor_bundle   = saved["anchor_bundle"]
            residual_bundle = saved["residual_bundle"]
            gate_weights    = saved["gate_weights"]
            gene_names      = saved["gene_names"]
            soma_names      = saved["soma_names"]
            x_scaler        = saved["x_scaler"]
            y_scaler        = saved["y_scaler"]
            trained = True
        else:
            print(f"Cached artifacts probe set differs from GNPC — retraining")

    if not trained:
        # Load all CKB training probe names first to define the intersection
        soma_all = pd.read_csv(DATA_DIR / "somascan_overlap_train.csv", nrows=0)
        ckb_probe_set = {c for c in soma_all.columns if c != "csid"}
        intersection = sorted(ckb_probe_set & gnpc_probe_set)

        print(f"Probe intersection: {len(intersection)} "
              f"(CKB {len(ckb_probe_set)}, GNPC QC'd {len(gnpc_probe_set)})")

        x_train, y_train, soma_names, gene_names, x_scaler, y_scaler = load_training_data(
            probe_subset=intersection,
        )
        anchor_bundle, residual_bundle, gate_weights = train_classical_aart(
            x_train, y_train, soma_names, gene_names
        )
        with ARTIFACTS_PATH.open("wb") as fh:
            pickle.dump({
                "anchor_bundle":   anchor_bundle,
                "residual_bundle": residual_bundle,
                "gate_weights":    gate_weights,
                "gene_names":      gene_names,
                "soma_names":      soma_names,
                "x_scaler":        x_scaler,
                "y_scaler":        y_scaler,
            }, fh)
        print(f"Artifacts saved to {ARTIFACTS_PATH}")

    # Apply AART to GNPC
    x_gnpc, sample_ids = load_gnpc_soma_aligned(gnpc_soma_path, soma_names, x_scaler)
    print(f"GNPC SomaScan preprocessed: {x_gnpc.shape}")

    print("Applying AART to GNPC...")
    anchor_gnpc   = anchor_bundle.predict(x_gnpc)
    residual_gnpc = residual_bundle.predict(x_gnpc)
    preds_std     = apply_gate(anchor_gnpc, residual_gnpc, gate_weights)

    preds_raw = y_scaler.inverse_transform(preds_std).astype(np.float32)

    gnpc_olink = pd.DataFrame(preds_raw, index=sample_ids, columns=gene_names)
    gnpc_olink.index.name = "sample_id"

    n_nan = gnpc_olink.isnull().sum().sum()
    if n_nan > 0:
        print(f"  Filling {n_nan} NaN values with column medians")
        gnpc_olink = gnpc_olink.fillna(gnpc_olink.median())

    gnpc_olink.to_parquet(out_path)
    print(f"\nSaved GNPC imputed Olink -> {out_path}")
    print(f"  Shape: {gnpc_olink.shape[0]} samples x {gnpc_olink.shape[1]} proteins")
    print(f"  Sample proteins: {gnpc_olink.columns[:5].tolist()}")


if __name__ == "__main__":
    main()
