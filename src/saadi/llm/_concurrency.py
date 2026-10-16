"""Small helper for sending a batch of independent requests concurrently."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_concurrently(fn: Callable[[T], R], items: Sequence[T], *, max_workers: int) -> list[R]:
    """Apply ``fn`` to every item in parallel threads, preserving input order.

    The first exception propagates to the caller. Items that have not started
    yet are cancelled at that point so a failing batch does not keep spending
    requests; items already running finish in the background.
    """

    items = list(items)
    if not items:
        return []
    workers = max(1, min(int(max_workers or 1), len(items)))
    if workers == 1:
        return [fn(item) for item in items]
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(fn, item) for item in items]
        results: list[R] = []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        return results
    finally:
        pool.shutdown(wait=True)
