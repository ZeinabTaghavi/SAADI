"""Loads Hugging Face tokenizers or processors for prompt formatting."""

from __future__ import annotations


def load_hf_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool = True,
    cache_dir: str | None = None,
    revision: str | None = None,
    loader: str = "tokenizer",
    extra_kwargs: dict[str, object] | None = None,
):
    try:
        from transformers import AutoProcessor, AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError("transformers is not installed. Install with `pip install .[transformers]`.") from exc
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
    }
    if revision:
        kwargs["revision"] = revision
    kwargs.update(dict(extra_kwargs or {}))

    normalized_loader = str(loader or "tokenizer").strip().lower()
    if normalized_loader == "tokenizer":
        formatter = AutoTokenizer.from_pretrained(model_id, **kwargs)
        tokenizer = formatter
    elif normalized_loader == "processor":
        formatter = AutoProcessor.from_pretrained(model_id, **kwargs)
        tokenizer = getattr(formatter, "tokenizer", None)
    else:
        raise ValueError(
            f"Unknown Hugging Face prompt formatter loader {loader!r}; "
            "expected 'tokenizer' or 'processor'"
        )

    if (
        tokenizer is not None
        and getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    return formatter
