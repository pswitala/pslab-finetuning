#!/usr/bin/env python3
"""Download statistical indicators from GUS BDL (Bank Danych Lokalnych) API.

GUS (Statistics Poland) data is public-domain / freely reusable. The BDL API returns
JSON time-series by subject + geographic unit. We turn each indicator into a short
Polish-language factual statement (verbalized) so the model can learn the facts as
text — this feeds synthetic QA in SFT and a small slice of CPT text.

API: https://bdl.stat.gov.pl/api/v1/
  GET /subjects                         -> subject tree (IDs like K11, K15, K27)
  GET /variables?subjectId=K11          -> variables under a subject (camelCase param)
  GET /data/by-variable/{varId}?...     -> values (by year, unit)

Subject IDs use alphanumeric codes (K11, K15, K27, K43 …). Run with
--list-subjects to print the top-level list before harvesting.

Usage:
    # discover subject IDs first:
    python scripts/ingest/gus_bdl.py --list-subjects

    # then harvest selected subjects:
    python scripts/ingest/gus_bdl.py --subjects K11,K15,K27 \
        --out data/catalogs/gus_bdl/indicators.jsonl --years 2018-2023
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.records import JsonlWriter, Record, get_json, today_iso  # noqa: E402

API = "https://bdl.stat.gov.pl/api/v1"


def verbalize(var_name: str, unit_name: str, year: int, value, measure: str) -> str:
    """Turn one data point into a clean Polish factual sentence."""
    return (f"Według danych GUS (BDL) wskaźnik „{var_name}” dla jednostki "
            f"„{unit_name}” w {year} roku wyniósł {value} {measure}.").strip()


def list_subjects(session: requests.Session) -> list[dict]:
    data = get_json(session, f"{API}/subjects",
                    params={"format": "json", "lang": "pl", "page-size": 100})
    return data.get("results", [])


def fetch_variables(session: requests.Session, subject_id: str,
                    max_vars: int = 0, delay: float = 0.5) -> list[dict]:
    """Paginate through variables for a subject with per-page throttling.

    max_vars=0 means unlimited. delay is seconds between page requests.
    BDL rate-limits hard above ~100 req/min — keep delay >= 0.5 s.
    """
    results = []
    page = 0
    while True:
        if page > 0:
            time.sleep(delay)
        data = get_json(session, f"{API}/variables",
                        params={"subjectId": subject_id, "format": "json",
                                "lang": "pl", "page-size": 100, "page": page})
        batch = data.get("results", [])
        results.extend(batch)
        if max_vars and len(results) >= max_vars:
            results = results[:max_vars]
            break
        if len(batch) < 100:
            break
        page += 1
    return results


def fetch_values(session: requests.Session, var_id: str, years: list[int],
                 delay: float = 0.3) -> list[dict]:
    time.sleep(delay)
    data = get_json(session, f"{API}/data/by-variable/{var_id}",
                    params={"format": "json", "lang": "pl", "page-size": 100,
                            "year": years})
    return data.get("results", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="",
                    help="comma-separated BDL subject IDs (e.g. K11,K15,K27). "
                         "Run --list-subjects to see available IDs.")
    ap.add_argument("--list-subjects", action="store_true",
                    help="print top-level subject IDs and names, then exit")
    ap.add_argument("--years", default="2018-2023")
    ap.add_argument("--out", default=None,
                    help="output jsonl path (required unless --list-subjects)")
    ap.add_argument("--max-vars-per-subject", type=int, default=500,
                    help="cap variables fetched per subject (0=unlimited). "
                         "K11 alone has 7700+ vars — keep low to avoid 429s. "
                         "Default: 500")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds between API requests (default 0.5)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if "-" in args.years:
        lo, hi = args.years.split("-", 1)
        years = list(range(int(lo), int(hi) + 1))
    else:
        years = [int(args.years)]

    session = requests.Session()

    if args.list_subjects:
        subjects = list_subjects(session)
        print(f"{'ID':<10} {'Name'}")
        print("-" * 50)
        for s in subjects:
            print(f"{s.get('id', ''):<10} {s.get('name', '')}")
        return 0

    if not args.subjects:
        ap.error("--subjects is required unless --list-subjects is used")
    if not args.out:
        ap.error("--out is required when harvesting data")

    n = 0
    with JsonlWriter(args.out) as w:
        for subject_id in args.subjects.split(","):
            subject_id = subject_id.strip()
            for var in fetch_variables(session, subject_id,
                                       max_vars=args.max_vars_per_subject,
                                       delay=args.delay):
                var_id = str(var.get("id"))
                var_name = var.get("n1") or var.get("name") or var_id
                measure = var.get("measureUnitName", "")
                try:
                    results = fetch_values(session, var_id, years, delay=args.delay)
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip var {var_id}: {exc}")
                    continue
                for unit in results:
                    unit_name = unit.get("name", "")
                    for point in unit.get("values", []):
                        yr, val = point.get("year"), point.get("val")
                        if yr is None or val is None:
                            continue
                        text = verbalize(var_name, unit_name, int(yr), val, measure)
                        w.write(Record(
                            id=f"gus_bdl:{var_id}:{unit.get('id')}:{yr}",
                            source="gus_bdl",
                            url=f"https://bdl.stat.gov.pl/bdl/dane/podgrup/wymiary/{var_id}",
                            license="public-domain",
                            snapshot_date=today_iso(),
                            title=var_name,
                            text=text,
                            lang="pl",
                            meta={"subject_id": subject_id, "var_id": var_id,
                                  "unit": unit_name, "year": yr, "value": val,
                                  "measure": measure},
                        ))
                        n += 1
                        if args.limit and n >= args.limit:
                            print(f"hit --limit {args.limit}")
                            return 0
    print(f"done: {n} indicator statements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
