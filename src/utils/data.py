from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitFrames:
    olink_train: pd.DataFrame
    olink_val: pd.DataFrame
    soma_train: pd.DataFrame
    soma_val: pd.DataFrame


def load_paired_tables(
    olink_csv: str | Path,
    soma_csv: str | Path,
    id_column: str = "csid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    olink = pd.read_csv(olink_csv)
    soma = pd.read_csv(soma_csv)
    if id_column not in olink.columns or id_column not in soma.columns:
        raise ValueError(f"Both input files must contain '{id_column}'")
    if len(olink) != len(soma):
        raise ValueError(f"Paired tables differ in row count: {len(olink)} vs {len(soma)}")
    if not np.array_equal(olink[id_column].astype(str).values, soma[id_column].astype(str).values):
        raise ValueError("Paired tables do not have identical sample order")
    return olink, soma


def split_external_train_for_validation(
    olink_train: pd.DataFrame,
    soma_train: pd.DataFrame,
    val_size: float,
    random_state: int,
    id_column: str = "csid",
) -> SplitFrames:
    if not np.array_equal(olink_train[id_column].astype(str).values, soma_train[id_column].astype(str).values):
        raise ValueError("Olink and SomaScan train tables are not aligned")
    indices = np.arange(len(olink_train))
    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=random_state)
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    return SplitFrames(
        olink_train=olink_train.iloc[train_idx].reset_index(drop=True),
        olink_val=olink_train.iloc[val_idx].reset_index(drop=True),
        soma_train=soma_train.iloc[train_idx].reset_index(drop=True),
        soma_val=soma_train.iloc[val_idx].reset_index(drop=True),
    )


def make_symlink(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source).resolve()
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.is_symlink() or target_path.exists():
        target_path.unlink()
    target_path.symlink_to(source_path)
    return target_path
