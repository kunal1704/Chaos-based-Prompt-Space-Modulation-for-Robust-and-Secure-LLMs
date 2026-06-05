# Geometry Experiment Suite

This suite implements the minimal reproducible geometry-only experiment requested in the research plan.

It intentionally excludes:

- Robustness evaluation.
- Prompt injection evaluation.
- Downstream task training or probing.

## Entry point

```powershell
conda run python experiments/run_geometry_suite.py
```

By default, the command uses `--embedding-source hash`, a deterministic local embedding source. This keeps the geometry pipeline runnable in environments where `torch` and `transformers` are unavailable.

To run on real Hugging Face input embeddings when dependencies and model weights are available:

```powershell
conda run python experiments/run_geometry_suite.py --embedding-source hf --model-name gpt2 --hash-dim 64
```

## Outputs

Default output directory:

```text
outputs/geometry/sample_hash_run/
```

Files:

- `config.json`: run configuration.
- `embeddings_manifest.jsonl`: prompt/token/shape metadata.
- `metrics.csv`: per-prompt, per-alpha, per-condition geometry metrics.
- `summary.json`: aggregate means and standard deviations.
- `run.log`: structured run log.
- `figures/*.png`: PCA and metric figures.

## Conditions

Each prompt is embedded, then compared under:

- `chaos`: ChaosFEX reduced to embedding width and scaled by `alpha`.
- `gaussian`: norm-matched Gaussian perturbation.
- `shuffled_chaos`: same ChaosFEX perturbation values with coordinates shuffled.
- `uniform`: norm-matched uniform perturbation.

The Gaussian and uniform baselines are Frobenius-norm matched to the chaos perturbation for each prompt and alpha.

## Metrics

- Mean token cosine similarity.
- Perturbation norm and relative perturbation norm.
- Pairwise token-distance distortion.
- Original/transformed anisotropy and anisotropy delta.
- Variance ratio.
- Linear CKA.
- Neighborhood preservation for configurable `k`.

## Interpretation

The committed sample run is a reproducibility smoke test over deterministic local embeddings. It validates that the experiment machinery, ChaosFEX wrapper, baselines, metrics, logging, and figures work end to end.

Scientific claims about GPT-2/BERT embeddings require rerunning with `--embedding-source hf` in an environment with `torch`, `transformers`, and model weights available.
