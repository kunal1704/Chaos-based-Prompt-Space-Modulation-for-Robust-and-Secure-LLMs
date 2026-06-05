"""Geometry and representation metrics for chaos-vs-noise comparisons."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


EPS = 1e-12


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denominator, EPS)


def mean_token_cosine(original: np.ndarray, transformed: np.ndarray) -> float:
    return float(np.mean(row_cosine(original, transformed)))


def pairwise_distance_distortion(original: np.ndarray, transformed: np.ndarray) -> float:
    if original.shape[0] < 2:
        return 0.0
    d0 = pdist(original, metric="euclidean")
    d1 = pdist(transformed, metric="euclidean")
    return float(np.mean(np.abs(d1 - d0) / np.maximum(d0, EPS)))


def anisotropy(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.maximum(norms, EPS)
    sim = normalized @ normalized.T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    return float(np.mean(sim[mask]))


def variance_ratio(original: np.ndarray, transformed: np.ndarray) -> float:
    base = float(np.var(original))
    changed = float(np.var(transformed))
    return changed / max(base, EPS)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    y_centered = y - np.mean(y, axis=0, keepdims=True)
    dot_xy = np.linalg.norm(x_centered.T @ y_centered, ord="fro") ** 2
    dot_xx = np.linalg.norm(x_centered.T @ x_centered, ord="fro")
    dot_yy = np.linalg.norm(y_centered.T @ y_centered, ord="fro")
    return float(dot_xy / max(dot_xx * dot_yy, EPS))


def neighborhood_preservation(original: np.ndarray, transformed: np.ndarray, k: int) -> float:
    n = original.shape[0]
    if n <= 1:
        return 1.0
    effective_k = min(k, n - 1)
    d0 = squareform(pdist(original, metric="euclidean"))
    d1 = squareform(pdist(transformed, metric="euclidean"))
    neighbors0 = np.argsort(d0, axis=1)[:, 1 : effective_k + 1]
    neighbors1 = np.argsort(d1, axis=1)[:, 1 : effective_k + 1]
    overlaps = [
        len(set(neighbors0[i]).intersection(neighbors1[i])) / effective_k
        for i in range(n)
    ]
    return float(np.mean(overlaps))


def summarize_pair(
    original: np.ndarray,
    transformed: np.ndarray,
    delta: np.ndarray,
    k_values: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    metrics = {
        "mean_token_cosine": mean_token_cosine(original, transformed),
        "perturbation_norm": float(np.linalg.norm(delta.ravel(), ord=2)),
        "relative_perturbation_norm": float(
            np.linalg.norm(delta.ravel(), ord=2) / max(np.linalg.norm(original.ravel(), ord=2), EPS)
        ),
        "pairwise_distance_distortion": pairwise_distance_distortion(original, transformed),
        "anisotropy_original": anisotropy(original),
        "anisotropy_transformed": anisotropy(transformed),
        "anisotropy_delta": anisotropy(transformed) - anisotropy(original),
        "variance_ratio": variance_ratio(original, transformed),
        "linear_cka": linear_cka(original, transformed),
    }
    for k in k_values:
        metrics[f"neighbor_preservation_k{k}"] = neighborhood_preservation(
            original, transformed, k
        )
    return metrics

