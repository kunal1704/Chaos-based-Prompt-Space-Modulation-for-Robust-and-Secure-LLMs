"""Baseline perturbations for geometry-only experiments."""

from __future__ import annotations

import numpy as np


def frobenius_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.ravel(), ord=2))


def match_frobenius_norm(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Scale candidate so its Frobenius norm matches target."""
    candidate_norm = frobenius_norm(candidate)
    target_norm = frobenius_norm(target)
    if candidate_norm == 0.0 or target_norm == 0.0:
        return np.zeros_like(candidate)
    return candidate * (target_norm / candidate_norm)


def gaussian_delta(shape: tuple[int, int], target_delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(loc=0.0, scale=1.0, size=shape)
    return match_frobenius_norm(raw, target_delta)


def shuffled_delta(target_delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = target_delta.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(target_delta.shape)


def uniform_delta(shape: tuple[int, int], target_delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = rng.uniform(low=-1.0, high=1.0, size=shape)
    return match_frobenius_norm(raw, target_delta)

