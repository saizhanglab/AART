from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingBundle:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    gene_names: list[str]
    soma_feature_names: list[str]
    id_column: str


def log_transform_soma_frame(df: pd.DataFrame, id_column: str = "csid") -> np.ndarray:
    values = df.drop(columns=[id_column]).astype(np.float32).to_numpy()
    if np.any(values <= 0):
        raise ValueError("SomaScan values must be strictly positive before log transform")
    return np.log(values).astype(np.float32)


def dataframe_targets(df: pd.DataFrame, id_column: str = "csid") -> np.ndarray:
    return df.drop(columns=[id_column]).astype(np.float32).to_numpy()


def dataframe_ids(df: pd.DataFrame, id_column: str = "csid") -> np.ndarray:
    return df[id_column].to_numpy()


def fit_preprocessing(
    olink_train: pd.DataFrame,
    soma_train: pd.DataFrame,
    id_column: str = "csid",
) -> tuple[PreprocessingBundle, np.ndarray, np.ndarray]:
    x_train_raw = log_transform_soma_frame(soma_train, id_column=id_column)
    y_train_raw = dataframe_targets(olink_train, id_column=id_column)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train_raw).astype(np.float32)
    y_train = y_scaler.fit_transform(y_train_raw).astype(np.float32)
    bundle = PreprocessingBundle(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        gene_names=[column for column in olink_train.columns if column != id_column],
        soma_feature_names=[column for column in soma_train.columns if column != id_column],
        id_column=id_column,
    )
    return bundle, x_train, y_train


def transform_soma(df: pd.DataFrame, bundle: PreprocessingBundle) -> np.ndarray:
    raw = log_transform_soma_frame(df, id_column=bundle.id_column)
    return bundle.x_scaler.transform(raw).astype(np.float32)


def transform_olink(df: pd.DataFrame, bundle: PreprocessingBundle) -> np.ndarray:
    raw = dataframe_targets(df, id_column=bundle.id_column)
    return bundle.y_scaler.transform(raw).astype(np.float32)


def inverse_transform_targets(values: np.ndarray, bundle: PreprocessingBundle) -> np.ndarray:
    return bundle.y_scaler.inverse_transform(values).astype(np.float32)


def bundle_to_serializable(bundle: PreprocessingBundle) -> dict[str, Any]:
    return {
        "gene_names": bundle.gene_names,
        "soma_feature_names": bundle.soma_feature_names,
        "id_column": bundle.id_column,
        "x_scaler_mean": bundle.x_scaler.mean_.tolist(),
        "x_scaler_scale": bundle.x_scaler.scale_.tolist(),
        "y_scaler_mean": bundle.y_scaler.mean_.tolist(),
        "y_scaler_scale": bundle.y_scaler.scale_.tolist(),
    }
