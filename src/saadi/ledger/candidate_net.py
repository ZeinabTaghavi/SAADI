"""The candidate net: a small open-type NER model lists every named span of a chunk.

Default model: ``urchade/gliner_large-v2.1`` with the labels
``[person, place, organization, named object, title]`` (spec, section D). The
GLiNER library truncates inputs at ``max_len`` words (384 for the large
models), so a 500-token chunk is split here into sentence-aligned windows of at
most ``window_words`` words and the spans are mapped back to chunk offsets.

Two implementations share one interface:

- :class:`CandidateNet`: GLiNER through the ``gliner`` package (torch). Loaded
  lazily on first use so that, inside the build process, it loads after vLLM
  has reserved its GPU memory.
- :class:`RegexCandidateNet`: capitalized-span heuristic with no model. Used by
  the unit tests and by ``--candidate-net regex`` dry runs; never by the GPU
  build unless explicitly configured.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "urchade/gliner_large-v2.1"
DEFAULT_LABELS = ("person", "place", "organization", "named object", "title")
DEFAULT_THRESHOLD = 0.5
DEFAULT_WINDOW_WORDS = 200
DEFAULT_MAX_CANDIDATES = 40
DEFAULT_MAX_SPAN_WORDS = 6

_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+")
_POSSESSIVE = re.compile(r"(?:'s|’s|')$", re.IGNORECASE)
_EDGE_PUNCT = "\"'“”‘’()[]{}<>«»,.;:!?-–—*_ "


@dataclass(slots=True)
class Candidate:
    """One named span proposed by the candidate net."""

    text: str
    label: str
    score: float
    start: int  # character offsets in the chunk text
    end: int
    sentence: str  # the sentence the span appears in (stored with a stub)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Candidate:
        return cls(
            text=str(row.get("text", "")),
            label=str(row.get("label", "")),
            score=float(row.get("score", 0.0) or 0.0),
            start=int(row.get("start", -1)),
            end=int(row.get("end", -1)),
            sentence=str(row.get("sentence", "")),
        )


@dataclass(slots=True)
class CandidateRun:
    """Candidates of one chunk plus what the filter removed and the time it took."""

    candidates: list[Candidate]
    dropped: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    raw_count: int = 0
    cap_hit: bool = False
    device: str = ""


# ---------------------------------------------------------------------------
# Sentences and windows
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of sentences, covering the text without gaps in order."""

    text = str(text or "")
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if end > pos:
            spans.append((pos, end))
        pos = match.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans or ([(0, len(text))] if text else [])


def sentence_windows(text: str, *, window_words: int) -> list[tuple[int, int]]:
    """Group consecutive sentences into windows of at most ``window_words`` words.

    A single sentence longer than the window is split on word boundaries.
    """

    windows: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_words = 0
    cur_end = 0
    for start, end in split_sentences(text):
        words = len(text[start:end].split())
        if words > window_words:
            if cur_start is not None:
                windows.append((cur_start, cur_end))
                cur_start, cur_words = None, 0
            # Split the long sentence itself.
            piece_start = start
            count = 0
            for m in re.finditer(r"\S+", text[start:end]):
                count += 1
                if count >= window_words:
                    piece_end = start + m.end()
                    windows.append((piece_start, piece_end))
                    piece_start = piece_end
                    count = 0
            if piece_start < end:
                windows.append((piece_start, end))
            continue
        if cur_start is None:
            cur_start, cur_words, cur_end = start, words, end
        elif cur_words + words <= window_words:
            cur_words += words
            cur_end = end
        else:
            windows.append((cur_start, cur_end))
            cur_start, cur_words, cur_end = start, words, end
    if cur_start is not None:
        windows.append((cur_start, cur_end))
    return windows


def sentence_of(text: str, start: int, end: int, sentences: list[tuple[int, int]] | None = None) -> str:
    sentences = sentences if sentences is not None else split_sentences(text)
    for s, e in sentences:
        if s <= start < e or s < end <= e:
            return text[s:e].strip()
    return text[max(0, start - 120) : min(len(text), end + 120)].strip()


# ---------------------------------------------------------------------------
# Span cleaning and filtering
# ---------------------------------------------------------------------------


def clean_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Trim edge punctuation/quotes and a trailing possessive; return the span and its new offsets."""

    span = text[start:end]
    left = len(span) - len(span.lstrip(_EDGE_PUNCT))
    span = span.lstrip(_EDGE_PUNCT)
    right_trim = len(span) - len(span.rstrip(_EDGE_PUNCT))
    span = span.rstrip(_EDGE_PUNCT)
    m = _POSSESSIVE.search(span)
    if m:
        span = span[: m.start()].rstrip()
        right_trim += end - start - left - len(span) - right_trim
    new_start = start + left
    new_end = new_start + len(span)
    return span, new_start, new_end


_LOWERCASE_ONLY_OK_LABELS = ()  # every candidate must carry a capital letter (named spans)


def _has_capital(text: str) -> bool:
    return any(ch.isalpha() and ch.isupper() for ch in text)


def filter_candidates(
    raw: Iterable[Candidate],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_span_words: int = DEFAULT_MAX_SPAN_WORDS,
) -> tuple[list[Candidate], list[dict[str, Any]], bool]:
    """Deduplicate and filter raw spans; return (kept, dropped, cap_hit).

    Rules (fixed, reported by the gate): a candidate keeps its first occurrence
    only (case-insensitive, whitespace-normalized); it must contain a letter and a
    capital letter; it has at most ``max_span_words`` words; at most
    ``max_candidates`` survive, highest score first, then chunk order.
    """

    seen: set[str] = set()
    kept: list[Candidate] = []
    dropped: list[dict[str, Any]] = []
    for cand in sorted(raw, key=lambda c: (c.start, c.end)):
        text = " ".join(cand.text.split())
        key = text.casefold()
        reason = None
        if not text or not any(ch.isalpha() for ch in text):
            reason = "no letters"
        elif not _has_capital(text):
            reason = "no capital letter"
        elif len(text.split()) > max_span_words:
            reason = f"longer than {max_span_words} words"
        elif len(text) > 80:
            reason = "longer than 80 characters"
        elif key in seen:
            reason = "duplicate surface"
        if reason:
            dropped.append({"text": text, "label": cand.label, "score": round(cand.score, 3), "reason": reason})
            continue
        seen.add(key)
        kept.append(Candidate(text=text, label=cand.label, score=cand.score, start=cand.start, end=cand.end, sentence=cand.sentence))
    cap_hit = False
    if len(kept) > max_candidates:
        cap_hit = True
        ranked = sorted(kept, key=lambda c: (-c.score, c.start))
        cut = ranked[max_candidates:]
        for c in cut:
            dropped.append(
                {"text": c.text, "label": c.label, "score": round(c.score, 3), "reason": f"beyond the cap of {max_candidates} candidates"}
            )
        keep_ids = {id(c) for c in ranked[:max_candidates]}
        kept = [c for c in kept if id(c) in keep_ids]
    return kept, dropped, cap_hit


# ---------------------------------------------------------------------------
# GLiNER
# ---------------------------------------------------------------------------


class CandidateNet:
    """GLiNER-based candidate net with lazy loading and sentence windows."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        labels: Iterable[str] = DEFAULT_LABELS,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "auto",
        window_words: int = DEFAULT_WINDOW_WORDS,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_span_words: int = DEFAULT_MAX_SPAN_WORDS,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.labels = [str(x) for x in labels]
        self.threshold = float(threshold)
        self.requested_device = str(device or "auto")
        self.window_words = int(window_words)
        self.max_candidates = int(max_candidates)
        self.max_span_words = int(max_span_words)
        self.local_files_only = local_files_only
        self._model = None
        self.device = ""
        self.load_seconds = 0.0
        self.versions: dict[str, str] = {}

    # -- loading -----------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        t0 = time.time()
        try:
            import torch
            from gliner import GLiNER
        except Exception as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "The candidate net needs the `gliner` package (and torch). Install it with "
                "`pip install --no-deps --target <dir> gliner` next to the study venv, or run sh/01_gate.sh."
            ) from exc
        import gliner as _gliner
        import transformers as _tf

        self.versions = {
            "torch": str(torch.__version__),
            "gliner": str(getattr(_gliner, "__version__", "?")),
            "transformers": str(_tf.__version__),
        }
        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"map_location": "cpu"}
        if self.local_files_only is not None:
            kwargs["local_files_only"] = bool(self.local_files_only)
        model = GLiNER.from_pretrained(self.model_name, **kwargs)
        model.eval()
        if device.startswith("cuda"):
            try:
                model = model.to(device)
            except Exception as exc:  # pragma: no cover - GPU only
                logger.warning("Candidate net could not move to %s (%s); using CPU.", device, exc)
                device = "cpu"
        self._model = model
        self.device = device
        self.load_seconds = time.time() - t0
        logger.info("Candidate net loaded model=%s device=%s in %.1fs", self.model_name, device, self.load_seconds)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- inference -----------------------------------------------------------
    def _predict_windows(self, windows: list[str]) -> list[list[dict[str, Any]]]:
        assert self._model is not None
        import torch

        with torch.no_grad():
            out = self._model.inference(windows, self.labels, flat_ner=True, threshold=self.threshold, batch_size=8)
        return out

    def run(self, text: str) -> CandidateRun:
        self.load()
        t0 = time.time()
        text = str(text or "")
        sentences = split_sentences(text)
        windows = sentence_windows(text, window_words=self.window_words)
        window_texts = [text[s:e] for s, e in windows]
        raw: list[Candidate] = []
        if window_texts:
            for (w_start, _w_end), spans in zip(windows, self._predict_windows(window_texts), strict=True):
                for span in spans:
                    start = w_start + int(span["start"])
                    end = w_start + int(span["end"])
                    surface, start, end = clean_span(text, start, end)
                    if not surface:
                        continue
                    raw.append(
                        Candidate(
                            text=surface,
                            label=str(span.get("label", "")),
                            score=float(span.get("score", 0.0) or 0.0),
                            start=start,
                            end=end,
                            sentence=sentence_of(text, start, end, sentences),
                        )
                    )
        kept, dropped, cap_hit = filter_candidates(raw, max_candidates=self.max_candidates, max_span_words=self.max_span_words)
        return CandidateRun(
            candidates=kept, dropped=dropped, seconds=time.time() - t0, raw_count=len(raw), cap_hit=cap_hit, device=self.device
        )

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "labels": list(self.labels),
            "threshold": self.threshold,
            "device": self.device or self.requested_device,
            "window_words": self.window_words,
            "max_candidates": self.max_candidates,
            "max_span_words": self.max_span_words,
            "versions": dict(self.versions),
        }


# ---------------------------------------------------------------------------
# Regex fallback (tests and dry runs only)
# ---------------------------------------------------------------------------

_CAP_SPAN = re.compile(r"(?:(?:the|The)\s+)?(?:[A-Z][\w'’-]*)(?:\s+(?:of|de|van|von|the|del|la|le)?\s*[A-Z][\w'’-]*)*")
_SENTENCE_START_WORDS = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "by",
    "he",
    "she",
    "it",
    "they",
    "we",
    "i",
    "you",
    "and",
    "but",
    "so",
    "then",
    "when",
    "well",
    "now",
    "there",
    "this",
    "that",
    "his",
    "her",
    "my",
    "our",
    "its",
    "if",
    "as",
    "for",
    "to",
    "of",
    "with",
    "from",
    "no",
    "yes",
    "oh",
    "what",
    "who",
    "where",
    "why",
    "how",
    "people",
    "everyone",
    "nobody",
    "one",
    "all",
    "every",
    "some",
    "after",
    "before",
    "during",
    "since",
    "while",
    "once",
    "here",
    "later",
    "soon",
    "still",
    "yet",
    "thus",
    "however",
}


class RegexCandidateNet:
    """Capitalized-span heuristic standing in for the model (tests and dry runs)."""

    model_name = "regex-capitalized-spans"
    device = "cpu"
    labels = ["span"]
    threshold = 1.0
    versions: dict[str, str] = {}
    load_seconds = 0.0
    loaded = True

    def __init__(self, *, max_candidates: int = DEFAULT_MAX_CANDIDATES, max_span_words: int = DEFAULT_MAX_SPAN_WORDS) -> None:
        self.max_candidates = int(max_candidates)
        self.max_span_words = int(max_span_words)

    def load(self) -> None:  # pragma: no cover - nothing to load
        return None

    def run(self, text: str) -> CandidateRun:
        t0 = time.time()
        text = str(text or "")
        sentences = split_sentences(text)
        raw: list[Candidate] = []
        for s, e in sentences:
            for m in _CAP_SPAN.finditer(text, s, e):
                surface, start, end = clean_span(text, m.start(), m.end())
                words = surface.split()
                # Drop a sentence-initial function word that is only capitalized by position.
                while words and words[0].lower() in _SENTENCE_START_WORDS and words[0] != "the" and words[0] != "The":
                    start += len(words[0]) + 1
                    words = words[1:]
                if words and words[0] in {"the", "The"} and len(words) == 1:
                    continue
                surface = " ".join(words)
                if not surface or not _has_capital(surface):
                    continue
                if surface.split()[0].lower() in _SENTENCE_START_WORDS and len(surface.split()) == 1:
                    continue
                raw.append(
                    Candidate(text=surface, label="span", score=1.0, start=start, end=start + len(surface), sentence=text[s:e].strip())
                )
        kept, dropped, cap_hit = filter_candidates(raw, max_candidates=self.max_candidates, max_span_words=self.max_span_words)
        return CandidateRun(candidates=kept, dropped=dropped, seconds=time.time() - t0, raw_count=len(raw), cap_hit=cap_hit, device="cpu")

    def describe(self) -> dict[str, Any]:
        return {"model": self.model_name, "labels": list(self.labels), "threshold": self.threshold, "device": "cpu", "versions": {}}


def build_candidate_net(cfg: dict[str, Any] | None) -> CandidateNet | RegexCandidateNet:
    """Build the candidate net from the profile's ``ledger`` mapping."""

    cfg = dict(cfg or {})
    kind = str(cfg.get("candidate_net", "gliner") or "gliner").strip().lower()
    max_candidates = int(cfg.get("candidate_max_per_chunk", DEFAULT_MAX_CANDIDATES))
    max_span_words = int(cfg.get("candidate_max_span_words", DEFAULT_MAX_SPAN_WORDS))
    if kind == "regex":
        return RegexCandidateNet(max_candidates=max_candidates, max_span_words=max_span_words)
    if kind != "gliner":
        raise ValueError(f"Unknown candidate_net {kind!r}; expected gliner or regex.")
    return CandidateNet(
        str(cfg.get("candidate_model", DEFAULT_MODEL)),
        list(cfg.get("candidate_labels") or DEFAULT_LABELS),
        threshold=float(cfg.get("candidate_threshold", DEFAULT_THRESHOLD)),
        device=str(cfg.get("candidate_device", "auto")),
        window_words=int(cfg.get("candidate_window_words", DEFAULT_WINDOW_WORDS)),
        max_candidates=max_candidates,
        max_span_words=max_span_words,
        local_files_only=cfg.get("candidate_local_files_only"),
    )


__all__ = [
    "Candidate",
    "CandidateNet",
    "CandidateRun",
    "DEFAULT_LABELS",
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
    "RegexCandidateNet",
    "build_candidate_net",
    "clean_span",
    "filter_candidates",
    "sentence_of",
    "sentence_windows",
    "split_sentences",
]
