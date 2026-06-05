"""Plotting helpers for geometry experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_metric_by_alpha(metrics: pd.DataFrame, out_path: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, group in metrics.groupby("condition"):
        agg = group.groupby("alpha")[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["alpha"],
            agg["mean"],
            yerr=agg["std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=condition,
        )
    ax.set_xlabel("alpha")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by modulation strength")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pca_spectrum(matrix: np.ndarray, out_path: Path, title: str) -> None:
    n_components = min(matrix.shape[0], matrix.shape[1], 20)
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, n_components + 1), pca.explained_variance_ratio_, marker="o")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pca_scatter(points: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    labels: list[str] = []
    matrices: list[np.ndarray] = []
    for label, matrix in points.items():
        labels.extend([label] * matrix.shape[0])
        matrices.append(matrix)
    combined = np.vstack(matrices)
    if combined.shape[0] < 2:
        return
    pca = PCA(n_components=2)
    xy = pca.fit_transform(combined)
    fig, ax = plt.subplots(figsize=(7, 6))
    labels_array = np.asarray(labels)
    for label in sorted(points):
        idx = labels_array == label
        ax.scatter(xy[idx, 0], xy[idx, 1], s=18, alpha=0.7, label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

