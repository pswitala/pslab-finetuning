"""Shared helpers for ingesting catalog data with provenance + license metadata.

Every ingested document is stored as one JSON line with a stable schema so that the
later processing pipeline can filter by license and trace any text back to its source
and snapshot date (the snapshot date defines the model's knowledge cutoff for facts
baked into the weights).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import requests


@dataclass
class Record:
    """One ingested document with provenance."""
    id: str                       # stable source id (e.g. "isap:DU/2020/1234")
    source: str                   # "isap" | "dane.gov.pl" | "gus_bdl" | ...
    url: str                      # canonical URL for the item
    license: str                  # e.g. "public-domain", "CC-BY-4.0", "CC0", "unknown"
    snapshot_date: str            # ISO date the data was fetched (knowledge cutoff)
    title: str = ""
    text: str = ""                # extracted plain text (may be empty for tabular)
    lang: str = "pl"
    meta: dict = field(default_factory=dict)  # source-specific extra fields

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class JsonlWriter:
    """Append Records to a jsonl file, creating parent dirs as needed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self.count = 0

    def __enter__(self) -> "JsonlWriter":
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def write(self, rec: Record) -> None:
        assert self._fh is not None
        self._fh.write(rec.to_json() + "\n")
        self.count += 1

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
        print(f"[JsonlWriter] wrote {self.count} records -> {self.path}")


def today_iso() -> str:
    """ISO date string for the current snapshot (knowledge cutoff stamp)."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def _retry_sleep(resp: "requests.Response | None", attempt: int, backoff: float) -> float:
    """Return how long to sleep before the next attempt.

    Honors Retry-After header on 429/503 (minimum 30 s). Otherwise uses
    exponential backoff, floored at 2 s so the first retry isn't instant.
    """
    if resp is not None and resp.status_code in (429, 503):
        after = resp.headers.get("Retry-After", "")
        try:
            return max(30.0, float(after))
        except (ValueError, TypeError):
            return 60.0
    return max(2.0, backoff ** attempt)


def get_json(session: requests.Session, url: str, *, params: dict | None = None,
             retries: int = 8, backoff: float = 2.0, timeout: int = 30) -> dict:
    """GET JSON with exponential backoff. 429/503 waits ≥ 30 s."""
    last_exc: Exception | None = None
    last_resp = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout,
                               headers={"User-Agent": "pslab-ingest/0.1"})
            last_resp = resp
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            sleep = _retry_sleep(last_resp, attempt, backoff)
            print(f"  [retry {attempt + 1}/{retries}] {url} failed ({exc}); "
                  f"sleeping {sleep:.1f}s")
            time.sleep(sleep)
            last_resp = None
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last_exc


def get_text(session: requests.Session, url: str, *, retries: int = 8,
             backoff: float = 2.0, timeout: int = 60) -> str:
    last_exc: Exception | None = None
    last_resp = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout,
                               headers={"User-Agent": "pslab-ingest/0.1"})
            last_resp = resp
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            sleep = _retry_sleep(last_resp, attempt, backoff)
            time.sleep(sleep)
            last_resp = None
    raise RuntimeError(f"GET text failed after {retries} tries: {url}") from last_exc


def batched(it: Iterable, n: int) -> Iterator[list]:
    """Yield lists of up to n items from an iterable."""
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
