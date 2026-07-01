#!/usr/bin/env python3
"""Download dataset metadata + textual resources from dane.gov.pl (national open-data).

dane.gov.pl exposes a JSON:API (and a CKAN-style API). Most datasets are CC-BY-4.0 or
CC0, but license varies PER DATASET — we capture it per record and let the processing
stage filter (commercial-safe = CC0 / CC-BY / public-domain only).

We harvest the rich Polish-language DESCRIPTIONS (dataset + resource notes), which are
clean prose useful for CPT and for grounding synthetic QA. Bulk tabular CSVs are
referenced via URL in meta but not downloaded here (handle selectively later).

API: https://api.dane.gov.pl/  (JSON:API, paginated)
  GET /1.4/datasets?page=N&per_page=100   -> datasets with attributes + license
VERIFY at execution: current API version prefix (/1.4 vs /doc), field names, and the
license field location — adjust selectors below if the schema differs.

Usage:
    python scripts/ingest/dane_gov.py --out data/catalogs/dane_gov/datasets.jsonl \
        --max-pages 50 --commercial-safe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.records import JsonlWriter, Record, get_json, today_iso  # noqa: E402

API = "https://api.dane.gov.pl/1.4/datasets"

COMMERCIAL_SAFE = {"cc0", "cc-by", "cc-by-4.0", "cc-by-3.0", "public-domain", "pddl"}


def normalize_license(raw) -> str:
    if not raw:
        return "unknown"
    r = str(raw).strip().lower().replace(" ", "-")
    return r


def is_commercial_safe(lic: str) -> bool:
    return any(lic.startswith(ok) for ok in COMMERCIAL_SAFE)


def harvest(out: str, max_pages: int, per_page: int, commercial_safe: bool) -> int:
    session = requests.Session()
    n = 0
    with JsonlWriter(out) as w:
        for page in range(1, max_pages + 1):
            data = get_json(session, API, params={"page": page, "per_page": per_page,
                                                  "lang": "pl"})
            items = data.get("data", [])
            if not items:
                print(f"[page {page}] empty — stopping")
                break
            for item in items:
                attrs = item.get("attributes", {})
                # VERIFY: license can live under attributes.license_chosen / license_id.
                lic = normalize_license(
                    attrs.get("license_chosen")
                    or attrs.get("license_id")
                    or attrs.get("license_name")
                )
                if commercial_safe and not is_commercial_safe(lic):
                    continue
                title = attrs.get("title", "")
                notes = attrs.get("notes", "") or ""
                text = f"{title}\n\n{notes}".strip()
                if len(text) < 80:
                    continue
                ds_id = item.get("id")
                w.write(Record(
                    id=f"dane.gov.pl:{ds_id}",
                    source="dane.gov.pl",
                    url=f"https://dane.gov.pl/pl/dataset/{ds_id}",
                    license=lic,
                    snapshot_date=today_iso(),
                    title=title,
                    text=text,
                    lang="pl",
                    meta={"resources_url": attrs.get("resources"),
                          "category": attrs.get("category"),
                          "update_frequency": attrs.get("update_frequency")},
                ))
                n += 1
            print(f"[page {page}] kept {n} so far")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--commercial-safe", action="store_true",
                    help="keep only CC0/CC-BY/public-domain datasets")
    args = ap.parse_args()
    total = harvest(args.out, args.max_pages, args.per_page, args.commercial_safe)
    print(f"done: {total} dataset records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
