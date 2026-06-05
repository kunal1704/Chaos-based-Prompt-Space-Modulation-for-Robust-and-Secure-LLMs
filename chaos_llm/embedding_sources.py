"""Embedding sources used by the geometry experiment suite."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class EmbeddedPrompt:
    prompt_id: str
    text: str
    tokens: list[str]
    embeddings: np.ndarray
    source: str
    model_name: str


class EmbeddingSource(Protocol):
    source_name: str
    model_name: str

    def embed(self, prompt_id: str, text: str) -> EmbeddedPrompt:
        ...


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


class HashEmbeddingSource:
    """Deterministic local embedding source for offline reproducible smoke runs."""

    source_name = "hash"

    def __init__(self, dim: int = 64, model_name: str = "hash-embedding-v1") -> None:
        self.dim = dim
        self.model_name = model_name

    def _token_vector(self, token: str, position: int) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(f"{token.lower()}::{position}::{self.dim}"))
        base = rng.normal(0.0, 1.0, self.dim)
        pos = np.sin((position + 1) / (np.arange(self.dim) + 1.0))
        return base + 0.05 * pos

    def embed(self, prompt_id: str, text: str) -> EmbeddedPrompt:
        tokens = simple_tokenize(text)
        if not tokens:
            tokens = ["<empty>"]
        matrix = np.vstack([self._token_vector(token, idx) for idx, token in enumerate(tokens)])
        return EmbeddedPrompt(
            prompt_id=prompt_id,
            text=text,
            tokens=tokens,
            embeddings=np.asarray(matrix, dtype=np.float64),
            source=self.source_name,
            model_name=self.model_name,
        )


class HuggingFaceEmbeddingSource:
    """Embedding source for real transformer input embeddings."""

    source_name = "hf"

    def __init__(self, model_name: str = "gpt2", device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The hf embedding source requires torch and transformers. "
                "Install them or run with --embedding-source hash."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def embed(self, prompt_id: str, text: str) -> EmbeddedPrompt:
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        with self.torch.no_grad():
            emb_layer = self.model.get_input_embeddings()
            emb = emb_layer(input_ids).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())
        return EmbeddedPrompt(
            prompt_id=prompt_id,
            text=text,
            tokens=tokens,
            embeddings=np.asarray(emb, dtype=np.float64),
            source=self.source_name,
            model_name=self.model_name,
        )


def make_embedding_source(
    source: str,
    model_name: str,
    hash_dim: int,
    device: str = "cpu",
) -> EmbeddingSource:
    if source == "hash":
        return HashEmbeddingSource(dim=hash_dim)
    if source == "hf":
        return HuggingFaceEmbeddingSource(model_name=model_name, device=device)
    raise ValueError(f"Unknown embedding source: {source}")

