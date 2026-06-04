from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_gene_histogram(metrics_long: pd.DataFrame, output_path: str | Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.violinplot(data=metrics_long, x="method", y="pearson_r", inner="quartile", ax=ax)
    ax.set_title("Gene-wise Pearson Distribution")
    ax.set_xlabel("Method")
    ax.set_ylabel("Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_scatter(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=frame, x=x_col, y=y_col, ax=ax, s=16, alpha=0.6)
    lower = min(frame[x_col].min(), frame[y_col].min())
    upper = max(frame[x_col].max(), frame[y_col].max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rankplot(frame: pd.DataFrame, delta_col: str, output_path: str | Path, title: str) -> None:
    sns.set_theme(style="whitegrid")
    ranked = frame.sort_values(delta_col).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ranked.index, ranked[delta_col], color="#3c5488")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Protein rank")
    ax.set_ylabel(delta_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history_df: pd.DataFrame, output_path: str | Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    if {"epoch", "train_loss"}.issubset(history_df.columns):
        subset = history_df.dropna(subset=["train_loss"])
        axes[0].plot(subset["epoch"], subset["train_loss"], label="train_loss")
    if {"epoch", "val_loss"}.issubset(history_df.columns):
        subset = history_df.dropna(subset=["val_loss"])
        axes[0].plot(subset["epoch"], subset["val_loss"], label="val_loss")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")
    axes[0].set_ylabel("Loss")
    if {"epoch", "val_gene_pearson_median"}.issubset(history_df.columns):
        subset = history_df.dropna(subset=["val_gene_pearson_median"])
        axes[1].plot(subset["epoch"], subset["val_gene_pearson_median"], label="val_gene_pearson_median")
    if {"epoch", "val_gene_pearson_mean"}.issubset(history_df.columns):
        subset = history_df.dropna(subset=["val_gene_pearson_mean"])
        axes[1].plot(subset["epoch"], subset["val_gene_pearson_mean"], label="val_gene_pearson_mean")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="best")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Pearson")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_representative_truth_vs_pred(
    *,
    truth_df: pd.DataFrame,
    pred_frames: dict[str, pd.DataFrame],
    representative_genes: list[str],
    output_path: str | Path,
) -> None:
    sns.set_theme(style="whitegrid")
    rows = len(pred_frames)
    cols = len(representative_genes)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    for row_index, (method_name, pred_df) in enumerate(pred_frames.items()):
        for col_index, gene in enumerate(representative_genes):
            ax = axes[row_index, col_index]
            ax.scatter(truth_df[gene], pred_df[gene], s=16, alpha=0.65)
            lower = min(truth_df[gene].min(), pred_df[gene].min())
            upper = max(truth_df[gene].max(), pred_df[gene].max())
            ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
            ax.set_title(f"{method_name}: {gene}")
            ax.set_xlabel("Truth")
            ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
