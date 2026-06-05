"""ChaosFEX transformation wrappers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import ChaosFEX.feature_extractor as CFX


@dataclass(frozen=True)
class ChaosConfig:
    initial_condition: float = 0.2
    threshold: float = 0.5
    epsilon: float = 0.01
    trajectory_length: int = 100

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x_min) / (x_max - x_min)


def chaos_features(embeddings: np.ndarray, config: ChaosConfig) -> np.ndarray:
    normalized = minmax_normalize(embeddings)
    transformed = CFX.transform(
        normalized,
        config.initial_condition,
        config.trajectory_length,
        config.epsilon,
        config.threshold,
    )
    if transformed is None:
        raise ValueError(f"ChaosFEX rejected inputs for config: {config}")
    return np.asarray(transformed, dtype=np.float64)


def reduce_chaos_features(features: np.ndarray, original_dim: int, method: str = "mean") -> np.ndarray:
    if features.shape[1] != original_dim * 4:
        raise ValueError(
            f"Expected ChaosFEX width {original_dim * 4}, got {features.shape[1]}"
        )
    grouped = features.reshape(features.shape[0], 4, original_dim)
    if method == "mean":
        return grouped.mean(axis=1)
    if method == "ttss":
        return grouped[:, 0, :]
    if method == "energy":
        return grouped[:, 1, :]
    if method == "time":
        return grouped[:, 2, :]
    if method == "entropy":
        return grouped[:, 3, :]
    raise ValueError(f"Unknown chaos reduction method: {method}")


def chaos_delta(
    embeddings: np.ndarray,
    alpha: float,
    config: ChaosConfig,
    reduction: str = "mean",
) -> np.ndarray:
    features = chaos_features(embeddings, config)
    reduced = reduce_chaos_features(features, embeddings.shape[1], method=reduction)
    return alpha * reduced

