from __future__ import annotations

import re
from typing import Any

import pandas as pd

CANONICAL_PROTEIN_RE = re.compile(r"[^A-Z0-9]+")
SEQ_ID_RE = re.compile(r"^seq\.(\d+)\.(\d+)$")


def canonicalize_symbol(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        return None
    canonical = CANONICAL_PROTEIN_RE.sub("_", text.upper()).strip("_")
    return canonical or None


def split_pipe_values(value: Any) -> list[str]:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        return []
    values: list[str] = []
    for item in text.split("|"):
        cleaned = item.strip()
        if cleaned and cleaned.lower() != "nan" and cleaned not in values:
            values.append(cleaned)
    return values


def seqid_to_probe_id(seqid_raw: str) -> str:
    match = SEQ_ID_RE.match(seqid_raw)
    if match is None:
        raise ValueError(f"Unsupported SomaScan seq id: {seqid_raw}")
    return f"{match.group(1)}-{match.group(2)}"


def build_candidate_table(
    gene_names: list[str],
    soma_feature_names: list[str],
    annotation_df: pd.DataFrame,
) -> pd.DataFrame:
    soma_map = pd.DataFrame({"apt_col": soma_feature_names})
    soma_map["probe_id"] = soma_map["apt_col"].map(seqid_to_probe_id)
    annotation = annotation_df.copy()
    annotation["probe_id"] = annotation["probe_id"].astype(str)
    soma_map = soma_map.merge(annotation, on="probe_id", how="left")
    soma_map["symbol_tokens"] = soma_map["symbol_values"].map(
        lambda value: [token for token in (canonicalize_symbol(item) for item in split_pipe_values(value)) if token]
    )
    soma_map["alias_tokens"] = soma_map["alias_values"].map(
        lambda value: [token for token in (canonicalize_symbol(item) for item in split_pipe_values(value)) if token]
    )
    soma_map["symbol_tokens"] = soma_map["symbol_tokens"].map(lambda values: values if isinstance(values, list) else [])
    soma_map["alias_tokens"] = soma_map["alias_tokens"].map(lambda values: values if isinstance(values, list) else [])

    gene_index = {name: idx for idx, name in enumerate(gene_names)}
    apt_index = {name: idx for idx, name in enumerate(soma_feature_names)}
    rows: list[dict[str, Any]] = []
    for gene in gene_names:
        protein_symbol = canonicalize_symbol(gene)
        if protein_symbol is None:
            continue
        direct_mask = soma_map["symbol_tokens"].map(lambda values: protein_symbol in values)
        alias_mask = soma_map["alias_tokens"].map(lambda values: protein_symbol in values)
        subset = soma_map.loc[direct_mask | alias_mask].copy()
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            rows.append(
                {
                    "gene": gene,
                    "protein_symbol": protein_symbol,
                    "gene_idx": gene_index[gene],
                    "apt_col": row["apt_col"],
                    "apt_idx": apt_index[row["apt_col"]],
                    "probe_id": row["probe_id"],
                    "match_type": "direct" if protein_symbol in row["symbol_tokens"] else "alias",
                    "symbol_values": row.get("symbol_values"),
                    "alias_values": row.get("alias_values"),
                    "genename_values": row.get("genename_values"),
                }
            )
    return pd.DataFrame(rows)
