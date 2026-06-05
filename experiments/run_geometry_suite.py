"""Run the minimal geometry-only chaos-vs-noise experiment suite."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from chaos_llm.baselines import gaussian_delta, shuffled_delta, uniform_delta
from chaos_llm.chaos import ChaosConfig, chaos_delta
from chaos_llm.embedding_sources import make_embedding_source
from chaos_llm.io import append_jsonl, ensure_dir, read_prompts, write_json
from chaos_llm.metrics import summarize_pair
from chaos_llm.plotting import plot_metric_by_alpha, plot_pca_scatter, plot_pca_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("experiments/prompts_geometry.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/geometry/sample_hash_run"))
    parser.add_argument("--embedding-source", choices=["hash", "hf"], default="hash")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--hash-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument("--initial-condition", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--trajectory-length", type=int, default=100)
    parser.add_argument("--reduction", default="mean")
    parser.add_argument("--seed", type=int, default=1704)
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 10])
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("geometry_suite")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def config_payload(args: argparse.Namespace, chaos_config: ChaosConfig) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_source": args.embedding_source,
        "model_name": args.model_name if args.embedding_source == "hf" else "hash-embedding-v1",
        "hash_dim": args.hash_dim,
        "alphas": args.alphas,
        "seed": args.seed,
        "reduction": args.reduction,
        "k_values": args.k_values,
        "chaos": chaos_config.to_dict(),
        "scope": "geometry_only_chaos_vs_noise",
        "excluded": ["robustness", "prompt_injection", "downstream_tasks"],
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    metric_cols = [
        "mean_token_cosine",
        "perturbation_norm",
        "relative_perturbation_norm",
        "pairwise_distance_distortion",
        "anisotropy_delta",
        "variance_ratio",
        "linear_cka",
    ]
    grouped = (
        metrics.groupby(["condition", "alpha"])[metric_cols]
        .agg(["mean", "std"])
        .round(8)
    )
    summary: dict[str, Any] = {"by_condition_alpha": {}}
    for index, row in grouped.iterrows():
        condition, alpha = index
        key = f"{condition}|alpha={alpha}"
        summary["by_condition_alpha"][key] = {
            f"{metric}_{stat}": float(row[(metric, stat)])
            for metric in metric_cols
            for stat in ["mean", "std"]
        }
    return summary


def main() -> None:
    args = parse_args()
    run_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(run_dir / "figures")
    logger = setup_logging(run_dir)

    chaos_config = ChaosConfig(
        initial_condition=args.initial_condition,
        threshold=args.threshold,
        epsilon=args.epsilon,
        trajectory_length=args.trajectory_length,
    )
    write_json(run_dir / "config.json", config_payload(args, chaos_config))

    prompts = read_prompts(args.prompts)
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts)
    source = make_embedding_source(args.embedding_source, args.model_name, args.hash_dim, args.device)

    rows: list[dict[str, Any]] = []
    pca_points_for_first_alpha: dict[str, list[np.ndarray]] = {
        "original": [],
        "chaos": [],
        "gaussian": [],
        "shuffled_chaos": [],
    }
    original_all: list[np.ndarray] = []

    for prompt_index, prompt in enumerate(prompts):
        prompt_id = f"p{prompt_index:03d}"
        embedded = source.embed(prompt_id, prompt)
        original = embedded.embeddings
        original_all.append(original)
        append_jsonl(
            run_dir / "embeddings_manifest.jsonl",
            {
                "prompt_id": prompt_id,
                "text": prompt,
                "tokens": embedded.tokens,
                "shape": list(original.shape),
                "source": embedded.source,
                "model_name": embedded.model_name,
            },
        )

        for alpha in args.alphas:
            rng = np.random.default_rng(args.seed + prompt_index * 1009 + int(alpha * 1_000_000))
            delta_chaos = chaos_delta(original, alpha, chaos_config, reduction=args.reduction)
            deltas = {
                "chaos": delta_chaos,
                "gaussian": gaussian_delta(original.shape, delta_chaos, rng),
                "shuffled_chaos": shuffled_delta(delta_chaos, rng),
                "uniform": uniform_delta(original.shape, delta_chaos, rng),
            }

            for condition, delta in deltas.items():
                transformed = original + delta
                metric_values = summarize_pair(
                    original,
                    transformed,
                    delta,
                    k_values=tuple(args.k_values),
                )
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "alpha": alpha,
                        "n_tokens": original.shape[0],
                        "dim": original.shape[1],
                        **metric_values,
                    }
                )
                if alpha == args.alphas[0] and condition in pca_points_for_first_alpha:
                    pca_points_for_first_alpha[condition].append(transformed)

            if alpha == args.alphas[0]:
                pca_points_for_first_alpha["original"].append(original)

        logger.info("Processed %s with shape %s", prompt_id, original.shape)

    write_metrics_csv(run_dir / "metrics.csv", rows)
    metrics_df = pd.DataFrame(rows)
    summary = aggregate_summary(metrics_df)
    write_json(run_dir / "summary.json", summary)

    for metric in ["mean_token_cosine", "pairwise_distance_distortion", "linear_cka", "anisotropy_delta"]:
        plot_metric_by_alpha(metrics_df, figure_dir / f"{metric}_by_alpha.png", metric)

    original_matrix = np.vstack(original_all)
    plot_pca_spectrum(
        original_matrix,
        figure_dir / "original_pca_spectrum.png",
        "Original embedding PCA spectrum",
    )

    scatter_points = {
        condition: np.vstack(matrices)
        for condition, matrices in pca_points_for_first_alpha.items()
        if matrices
    }
    plot_pca_scatter(
        scatter_points,
        figure_dir / "pca_scatter_first_alpha.png",
        f"PCA scatter at alpha={args.alphas[0]}",
    )

    logger.info("Wrote metrics to %s", run_dir / "metrics.csv")
    logger.info("Wrote summary to %s", run_dir / "summary.json")
    logger.info("Wrote figures to %s", figure_dir)


if __name__ == "__main__":
    main()
