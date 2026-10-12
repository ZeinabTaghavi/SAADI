"""Generic Hugging Face embedding retriever helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class HFEmbeddingRetriever:
    """Dense retriever backed by a Hugging Face encoder model.

    Subclasses provide the default model name and any query/document prefixes
    required by a specific embedding family.
    """

    corpus: list[str]
    chunk_ids: list[str] | None = None
    k: int = 10
    model_name: str = ""
    device: str | None = None
    batch_size: int = 64
    max_length: int = 512
    trust_remote_code: bool = False
    pooling: str = "auto"
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""

    ENV_PREFIX: ClassVar[str] = "SAADI_EMBED"
    _MODEL_CACHE: ClassVar[dict[tuple[str, str, str, bool], tuple[Any, Any]]] = {}

    def __post_init__(self):
        self.chunk_ids = self.chunk_ids or [str(i) for i in range(len(self.corpus))]
        if len(self.chunk_ids) != len(self.corpus):
            raise ValueError("chunk_ids length must match corpus length")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique")

        self.model_name = self._resolve_string_env(f"{self.ENV_PREFIX}_MODEL", self.model_name)
        self.batch_size = self._resolve_int_env(f"{self.ENV_PREFIX}_BATCH_SIZE", self.batch_size, minimum=1)
        self.max_length = self._resolve_int_env(f"{self.ENV_PREFIX}_MAX_LENGTH", self.max_length, minimum=1)
        self._log_every_batches = self._resolve_int_env(
            f"{self.ENV_PREFIX}_LOG_EVERY_BATCHES",
            self._resolve_int_env("SAADI_RETRIEVER_LOG_EVERY_BATCHES", 10, minimum=1),
            minimum=1,
        )
        self.trust_remote_code = self._resolve_bool_env(
            f"{self.ENV_PREFIX}_TRUST_REMOTE_CODE",
            self.trust_remote_code,
        )
        self.pooling = (
            self._resolve_string_env(f"{self.ENV_PREFIX}_POOLING", self.pooling or "auto").strip().lower() or "auto"
        )
        self.query_prefix = self._resolve_string_env(f"{self.ENV_PREFIX}_QUERY_PREFIX", self.query_prefix)
        self.document_prefix = self._resolve_string_env(f"{self.ENV_PREFIX}_DOCUMENT_PREFIX", self.document_prefix)
        self.normalize_embeddings = self._resolve_bool_env(
            f"{self.ENV_PREFIX}_NORMALIZE",
            self.normalize_embeddings,
        )

        requested_device = (
            self.device
            or self._resolve_string_env(f"{self.ENV_PREFIX}_DEVICE", "")
            or self._resolve_string_env("SAADI_RETRIEVER_DEVICE", "")
            or None
        )

        self._torch = self._load_torch()
        requested_device_map = self._resolve_string_env(
            f"{self.ENV_PREFIX}_DEVICE_MAP",
            self._resolve_string_env("SAADI_RETRIEVER_DEVICE_MAP", ""),
        )
        logger.info(
            "Loading dense retriever model=%s env_prefix=%s requested_device=%s requested_device_map=%s batch_size=%d max_length=%d corpus_size=%d",
            self.model_name,
            self.ENV_PREFIX,
            str(requested_device or "") or "<auto>",
            str(requested_device_map or "") or "<none>",
            int(self.batch_size),
            int(self.max_length),
            len(self.corpus),
        )
        self._tokenizer, self._model, self.device, self._device_map = self._load_model(
            requested_device=requested_device,
            requested_device_map=requested_device_map,
        )
        logger.info(
            "Dense retriever ready model=%s runtime_device=%s device_map=%s",
            self.model_name,
            self.device,
            self._device_map or "<none>",
        )
        self._corpus_embeddings = self._encode_texts(self.corpus, text_kind="document")

    @staticmethod
    def _resolve_string_env(name: str, default: str) -> str:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = str(raw).strip()
        return value if value else default

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
    def _resolve_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None or not str(raw).strip():
            return bool(default)
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean-like value (got: {raw})")

    @staticmethod
    def _load_torch():
        try:
            import torch  # type: ignore

            return torch
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrievers require PyTorch. Install torch and transformers, or use retriever=bm25."
            ) from exc

    def _load_model(
        self,
        *,
        requested_device: str | None,
        requested_device_map: str,
    ) -> tuple[Any, Any, str, str]:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrievers require transformers. Install torch and transformers, or use retriever=bm25."
            ) from exc

        resolved_device = str(requested_device or "").strip()
        resolved_device_map = str(requested_device_map or "").strip()
        if resolved_device_map:
            runtime_device = "cpu"
        else:
            if not resolved_device:
                resolved_device = "cuda" if self._torch.cuda.is_available() else "cpu"
            elif resolved_device.startswith("cuda") and not self._torch.cuda.is_available():
                resolved_device = "cpu"
            runtime_device = resolved_device

        cache_key = (
            str(self.model_name),
            runtime_device,
            resolved_device_map,
            bool(self.trust_remote_code),
        )
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            tokenizer, model = cached
            logger.info(
                "Reusing cached dense retriever model=%s runtime_device=%s device_map=%s",
                self.model_name,
                runtime_device,
                resolved_device_map or "<none>",
            )
            return tokenizer, model, runtime_device, resolved_device_map

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if resolved_device_map:
            model_kwargs["device_map"] = resolved_device_map
        logger.info(
            "Instantiating dense retriever model=%s runtime_device=%s device_map=%s trust_remote_code=%s",
            self.model_name,
            runtime_device,
            resolved_device_map or "<none>",
            bool(self.trust_remote_code),
        )
        model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        if not resolved_device_map:
            model.to(runtime_device)
        model.eval()

        self._MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model, runtime_device, resolved_device_map

    def _prepare_texts(self, texts: list[str], *, text_kind: str) -> list[str]:
        prefix = self.query_prefix if text_kind == "query" else self.document_prefix
        if not prefix:
            return [str(text or "") for text in texts]
        return [f"{prefix}{str(text or '')}" for text in texts]

    def _mean_pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _last_token_pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        lengths = attention_mask.sum(dim=1).clamp(min=1).to(dtype=self._torch.long) - 1
        gather_index = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
        return last_hidden_state.gather(dim=1, index=gather_index).squeeze(1)

    def _resolve_pooling_mode(self, model_output: Any) -> str:
        mode = str(self.pooling or "auto").strip().lower() or "auto"
        if mode != "auto":
            return mode

        config = getattr(self._model, "config", None)
        for attr_name in ("sentence_pooling_method", "pooling_type", "pooler_type"):
            value = getattr(config, attr_name, None)
            if value:
                resolved = str(value).strip().lower()
                if resolved in {"mean", "avg"}:
                    return "mean"
                if resolved in {"cls", "cls_token"}:
                    return "cls"
                if resolved in {"last", "last_token"}:
                    return "last_token"

        if getattr(model_output, "pooler_output", None) is not None:
            return "pooler_output"
        return "mean"

    def _pool_embeddings(self, model_output: Any, *, attention_mask: Any) -> Any:
        if getattr(model_output, "sentence_embeddings", None) is not None:
            pooled = model_output.sentence_embeddings
        elif getattr(model_output, "embeddings", None) is not None:
            pooled = model_output.embeddings
        else:
            last_hidden_state = getattr(model_output, "last_hidden_state", None)
            if last_hidden_state is None and isinstance(model_output, tuple) and model_output:
                last_hidden_state = model_output[0]
            if last_hidden_state is None:
                raise RuntimeError(
                    f"Could not extract embeddings from model output for model={self.model_name!r}"
                )

            mode = self._resolve_pooling_mode(model_output)
            if mode == "cls":
                pooled = last_hidden_state[:, 0]
            elif mode == "last_token":
                pooled = self._last_token_pool(last_hidden_state, attention_mask)
            elif mode == "pooler_output":
                pooled = model_output.pooler_output
            else:
                pooled = self._mean_pool(last_hidden_state, attention_mask)

        if self.normalize_embeddings:
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    def _model_input_device(self) -> Any:
        get_input_embeddings = getattr(self._model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            try:
                embeddings = get_input_embeddings()
            except (AttributeError, NotImplementedError):
                embeddings = None
            if embeddings is not None:
                weight = getattr(embeddings, "weight", None)
                device = getattr(weight, "device", None)
                if device is not None and str(device) != "meta":
                    return device

        for param in self._model.parameters():
            device = getattr(param, "device", None)
            if device is not None and str(device) != "meta":
                return device
        return self.device

    def _encode_texts(self, texts: list[str], *, text_kind: str = "document") -> Any:
        if not texts:
            hidden = int(getattr(getattr(self._model, "config", None), "hidden_size", 0) or 0)
            return self._torch.empty((0, hidden), dtype=self._torch.float32)

        prepared = self._prepare_texts(list(texts), text_kind=text_kind)
        outputs: list[Any] = []
        input_device = self._model_input_device()
        total_batches = (len(prepared) + self.batch_size - 1) // self.batch_size
        should_log_progress = str(text_kind) == "document" and len(prepared) > self.batch_size
        started_at = time.perf_counter()
        if should_log_progress:
            logger.info(
                "Encoding document corpus model=%s texts=%d batches=%d batch_size=%d input_device=%s",
                self.model_name,
                len(prepared),
                total_batches,
                int(self.batch_size),
                str(input_device),
            )
        with self._torch.no_grad():
            for batch_idx, start in enumerate(range(0, len(prepared), self.batch_size), start=1):
                batch = prepared[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(input_device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                model_output = self._model(**encoded)
                pooled = self._pool_embeddings(
                    model_output,
                    attention_mask=encoded["attention_mask"],
                )
                outputs.append(pooled.cpu())
                if should_log_progress and (
                    batch_idx == 1
                    or batch_idx == total_batches
                    or batch_idx % int(self._log_every_batches) == 0
                ):
                    logger.info(
                        "Dense retriever encoding progress model=%s batch=%d/%d texts=%d/%d elapsed=%.1fs",
                        self.model_name,
                        batch_idx,
                        total_batches,
                        min(start + len(batch), len(prepared)),
                        len(prepared),
                        time.perf_counter() - started_at,
                    )
        embeddings = self._torch.cat(outputs, dim=0)
        if should_log_progress:
            logger.info(
                "Completed document corpus encoding model=%s texts=%d batches=%d elapsed=%.1fs",
                self.model_name,
                len(prepared),
                total_batches,
                time.perf_counter() - started_at,
            )
        return embeddings

    def retrieve(self, query: str, *, k: int | None = None) -> tuple[list[str], list[float], list[int]]:
        if not self.corpus:
            return [], [], []
        topk = min(int(k or self.k), len(self.corpus))
        if topk <= 0:
            return [], [], []

        query_embedding = self._encode_texts([query], text_kind="query").squeeze(0)
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


__all__ = ["HFEmbeddingRetriever"]
