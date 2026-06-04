from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .metrics import compute_prediction_summary
from .mapping import canonicalize_symbol


@dataclass
class AnchorBundle:
    gene_names: list[str]
    models: list[Ridge | None]
    feature_indices: list[list[int]]
    feature_names: list[list[str]]
    match_types: list[str]
    train_anchor_corr: np.ndarray

    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        y_train: np.ndarray,
        gene_names: list[str],
        candidates: pd.DataFrame,
        alpha_single: float,
        alpha_multi: float,
    ) -> "AnchorBundle":
        n_targets = len(gene_names)
        models: list[Ridge | None] = [None] * n_targets
        feature_indices: list[list[int]] = [[] for _ in range(n_targets)]
        feature_names: list[list[str]] = [[] for _ in range(n_targets)]
        match_types: list[str] = ["none"] * n_targets
        train_anchor_corr = np.zeros(n_targets, dtype=np.float32)

        if candidates.empty:
            return cls(gene_names, models, feature_indices, feature_names, match_types, train_anchor_corr)

        for gene, group in candidates.groupby("gene", sort=False):
            gene_idx = int(group["gene_idx"].iloc[0])
            direct = group[group["match_type"] == "direct"]
            use = direct if not direct.empty else group
            indices = use["apt_idx"].astype(int).tolist()
            alpha = alpha_multi if len(indices) > 1 else alpha_single
            model = Ridge(alpha=alpha)
            model.fit(x_train[:, indices], y_train[:, gene_idx])
            predictions = model.predict(x_train[:, indices]).astype(np.float32)
            corr = np.corrcoef(predictions, y_train[:, gene_idx])[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
            models[gene_idx] = model
            feature_indices[gene_idx] = indices
            feature_names[gene_idx] = use["apt_col"].tolist()
            match_types[gene_idx] = "direct" if not direct.empty else "alias"
            train_anchor_corr[gene_idx] = float(corr)

        return cls(gene_names, models, feature_indices, feature_names, match_types, train_anchor_corr)

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.zeros((x.shape[0], len(self.gene_names)), dtype=np.float32)
        for gene_idx, model in enumerate(self.models):
            if model is None:
                continue
            predictions[:, gene_idx] = model.predict(x[:, self.feature_indices[gene_idx]]).astype(np.float32)
        return predictions

    def metadata_frame(self) -> pd.DataFrame:
        rows = []
        for gene_idx, gene in enumerate(self.gene_names):
            rows.append(
                {
                    "gene": gene,
                    "protein_symbol": canonicalize_symbol(gene) or gene,
                    "gene_idx": gene_idx,
                    "match_type": self.match_types[gene_idx],
                    "n_candidates_used": len(self.feature_indices[gene_idx]),
                    "anchor_features": "|".join(self.feature_names[gene_idx]),
                    "train_anchor_corr": float(self.train_anchor_corr[gene_idx]),
                    "has_anchor": float(self.models[gene_idx] is not None),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class ResidualBundle:
    pca: PCA
    ridge: Ridge

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ridge.predict(self.pca.transform(x)).astype(np.float32)


def fit_residual_bundle(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    n_components: int,
    alpha: float,
    random_state: int,
) -> ResidualBundle:
    max_components = max(1, min(n_components, x_train.shape[0] - 1, x_train.shape[1]))
    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=random_state)
    z_train = pca.fit_transform(x_train).astype(np.float32)
    ridge = Ridge(alpha=alpha)
    ridge.fit(z_train, residual_train)
    return ResidualBundle(pca=pca, ridge=ridge)


def compute_residual_prior(anchor_bundle: AnchorBundle) -> np.ndarray:
    prior = 1.0 - np.clip(np.maximum(anchor_bundle.train_anchor_corr, 0.0), 0.05, 0.95)
    no_anchor_mask = np.array([model is None for model in anchor_bundle.models], dtype=bool)
    prior[no_anchor_mask] = 0.95
    return prior.astype(np.float32)


def fit_closed_form_gate(
    residual_val_pred: np.ndarray,
    anchor_val: np.ndarray,
    y_val: np.ndarray,
    residual_prior: np.ndarray,
    lam: float,
) -> np.ndarray:
    numerator = (residual_val_pred * (y_val - anchor_val)).sum(axis=0) + lam * residual_prior
    denominator = np.square(residual_val_pred).sum(axis=0) + lam
    gate = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def apply_gate(anchor: np.ndarray, residual_pred: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return anchor + residual_pred * gate.reshape(1, -1)


def summarize_baseline_method(
    method_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    validation_score: float,
) -> dict[str, float | str]:
    return {"method": method_name, "validation_gene_pearson_median": validation_score, **compute_prediction_summary(y_true, y_pred)}
