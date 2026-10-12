"""Dense retriever adapter built on top of the Contriever encoder."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, ClassVar


@dataclass
class ContrieverRetriever:
    corpus: list[str]
    chunk_ids: list[str] | None = None
    k: int = 10
    model_name: str = "facebook/contriever"
    device: str | None = None
    batch_size: int = 128
    max_length: int = 512

    _MODEL_CACHE: ClassVar[dict[tuple[str, str], tuple[Any, Any]]] = {}

    def __post_init__(self):
        self.chunk_ids = self.chunk_ids or [str(i) for i in range(len(self.corpus))]
        if len(self.chunk_ids) != len(self.corpus):
            raise ValueError("chunk_ids length must match corpus length")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique")

        env_model = os.getenv("SAADI_CONTRIEVER_MODEL", self.model_name).strip() or self.model_name
        env_device = os.getenv("SAADI_RETRIEVER_DEVICE", "").strip()
        resolved_device = self.device or env_device or None
        self.model_name = env_model
        self.batch_size = self._resolve_int_env("SAADI_CONTRIEVER_BATCH_SIZE", self.batch_size, minimum=1)
        self.max_length = self._resolve_int_env("SAADI_CONTRIEVER_MAX_LENGTH", self.max_length, minimum=1)

        self._torch = self._load_torch()
        self._tokenizer, self._model, self.device = self._load_model(resolved_device)
        self._corpus_embeddings = self._encode_texts(self.corpus)

    @staticmethod
    def _resolve_int_env(name: str, default: int, *, minimum: int) -> int:
        raw = os.getenv(name)
        if raw is None or not str(raw).strip():
            return default
        try:
            value = int(raw)
        except Exception as exc:
            raise ValueError(f"{name} must be an integer (got: {raw})") from exc
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum} (got: {value})")
        return value

    @staticmethod
    def _load_torch():
        try:
            import torch  # type: ignore
            return torch
        except ImportError as exc:
            raise RuntimeError(
                "Contriever retriever requires PyTorch. Install torch and transformers, or use retriever=bm25."
            ) from exc

    def _load_model(self, requested_device: str | None) -> tuple[Any, Any, str]:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Contriever retriever requires transformers. Install torch and transformers, or use retriever=bm25."
            ) from exc

        resolved_device = (requested_device or "").strip()
        if not resolved_device:
            resolved_device = "cuda" if self._torch.cuda.is_available() else "cpu"
        elif resolved_device.startswith("cuda") and not self._torch.cuda.is_available():
            resolved_device = "cpu"

        cache_key = (self.model_name, resolved_device)
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            tokenizer, model = cached
            return tokenizer, model, resolved_device

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # facebook/contriever ships pytorch_model.bin on `main`; the only
        # safetensors copy lives in a community PR (refs/pr/2). Letting
        # transformers auto-prefer safetensors makes it hit the HF CDN to
        # resolve that PR, which can hang indefinitely on restricted nodes.
        # Force the .bin weights so loading stays on the cached main revision.
        model = AutoModel.from_pretrained(self.model_name, use_safetensors=False)
        model.to(resolved_device)
        model.eval()
        self._MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model, resolved_device

    def _mean_pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return self._torch.nn.functional.normalize(pooled, p=2, dim=1)

    def _encode_texts(self, texts: list[str]) -> Any:
        if not texts:
            hidden = int(getattr(getattr(self._model, "config", None), "hidden_size", 0) or 0)
            return self._torch.empty((0, hidden), dtype=self._torch.float32)

        outputs: list[Any] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                model_output = self._model(**encoded)
                pooled = self._mean_pool(model_output.last_hidden_state, encoded["attention_mask"])
                outputs.append(pooled.cpu())
        return self._torch.cat(outputs, dim=0)

    def retrieve(self, query: str, *, k: int | None = None) -> tuple[list[str], list[float], list[int]]:
        if not self.corpus:
            return [], [], []
        topk = min(int(k or self.k), len(self.corpus))
        if topk <= 0:
            return [], [], []

        query_embedding = self._encode_texts([query]).squeeze(0)
        scores = self._corpus_embeddings @ query_embedding
        top_scores, top_indices = self._torch.topk(scores, topk, largest=True, sorted=True)
        idxs = [int(i) for i in top_indices.tolist()]
        return (
            [self.chunk_ids[i] for i in idxs],
            [float(s) for s in top_scores.tolist()],
            idxs,
        )

    def retrieve_data(self, query: str):
        ids, scores, idxs = self.retrieve(query)
        texts = [self.corpus[i] for i in idxs]
        return texts, scores
