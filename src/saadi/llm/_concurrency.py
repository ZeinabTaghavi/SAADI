"""Small helper for sending a batch of independent requests concurrently."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_concurrently(fn: Callable[[T], R], items: Sequence[T], *, max_workers: int) -> list[R]:
    """Apply ``fn`` to every item, in parallel threads, preserving input order.

    The first exception raised by any item propagates to the caller after the
    pool has drained.
    """

    items = list(items)
    if not items:
        return []
    workers = max(1, min(int(max_workers or 1), len(items)))
    if workers == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
