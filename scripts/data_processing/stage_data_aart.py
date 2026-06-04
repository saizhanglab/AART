#!/usr/bin/env python3
"""
Prepare paired proteomics data for AART training.

  1. Split raw paired data into train/test
  2. Build a SomaScan/MS annotation table (probe_id -> gene symbol mapping)

Supports three platform pairs:
  - CKB:  SomaScan <-> Olink   (annotation from Wang et al. Excel)
  - PEX_LC: MS <-> Olink       (annotation from MS feature metadata)
  - GNPC: MS <-> SomaScan      (annotation from MS feature metadata)

Usage:
  python stage_data_aart.py --config configs/aart/soma_to_olink.yaml
  python stage_data_aart.py --config configs/aart/ms_to_olink.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aart.mapping import seqid_to_probe_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split paired data and build annotation table")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "aart" / "base.yaml"))
    return parser.parse_args()


def build_annotation_from_excel(excel_path: Path) -> pd.DataFrame:
    """Build annotation from Wang et al. Supplementary Data 1 (Olink <-> SomaScan)."""
    mapping = pd.read_excel(excel_path, sheet_name="Supplementary_Data_1")
    mapping["probe_id"] = mapping["somascan_id"].apply(seqid_to_probe_id)
    return (
        mapping
        .groupby("probe_id", sort=False)
        .agg(
            symbol_values=("olink_id", lambda s: "|".join(s.dropna().unique())),
            alias_values=("olink_id", lambda s: "|".join(s.dropna().unique())),
            genename_values=("uniprot_id", lambda s: "|".join(s.dropna().unique())),
        )
        .reset_index()
    )


def build_annotation_from_feature_metadata(metadata_path: Path) -> pd.DataFrame:
    """Build annotation from MS feature metadata CSV (probe_id, gene_name, uniprot columns)."""
    meta = pd.read_csv(metadata_path)
    return pd.DataFrame({
        "probe_id": meta["probe_id"].astype(str),
        "symbol_values": meta["gene_name"].fillna(""),
        "alias_values": meta["gene_name"].fillna(""),
        "genename_values": meta.get("description", meta.get("uniprot", pd.Series("", index=meta.index))).fillna(""),
    })


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    split_config = config["split"]
    data_dir = PROJECT_ROOT / Path(config["paths"]["data_dir"])
    annotation_config = config.get("annotation", {})

    # --- Split paired data ---
    source_input = config["paths"].get("source_input", "somascan_overlap.csv")
    source_target = config["paths"].get("source_target", "olink_overlap.csv")

    input_df = pd.read_csv(data_dir / source_input)
    target_df = pd.read_csv(data_dir / source_target)
    assert len(input_df) == len(target_df), "Row count mismatch between input and target"

    rng = np.random.RandomState(split_config["random_state"])
    n = len(input_df)
    test_size = split_config.get("test_size", 0.2)
    perm = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])

    train_input_name = Path(config["paths"]["train_soma"]).name
    test_input_name = Path(config["paths"]["test_soma"]).name
    train_target_name = Path(config["paths"]["train_olink"]).name
    test_target_name = Path(config["paths"]["test_olink"]).name

    input_df.iloc[train_idx].to_csv(data_dir / train_input_name, index=False)
    input_df.iloc[test_idx].to_csv(data_dir / test_input_name, index=False)
    target_df.iloc[train_idx].to_csv(data_dir / train_target_name, index=False)
    target_df.iloc[test_idx].to_csv(data_dir / test_target_name, index=False)
    print(f"Split: {len(train_idx)} train, {len(test_idx)} test -> {data_dir}")

    # --- Build annotation ---
    annotation_source = annotation_config.get("source", "excel")
    annotation_file = annotation_config.get("file")

    if annotation_source == "excel" and annotation_file:
        annotation = build_annotation_from_excel(data_dir / annotation_file)
    elif annotation_source == "feature_metadata" and annotation_file:
        annotation = build_annotation_from_feature_metadata(data_dir / annotation_file)
    else:
        print("No annotation source specified, skipping annotation build.")
        return

    annotation_out = data_dir / Path(config["paths"]["annotation"]).name
    annotation.to_csv(annotation_out, index=False)
    print(f"Annotation: {len(annotation)} probes -> {annotation_out}")
    print("Done.")


if __name__ == "__main__":
    main()
