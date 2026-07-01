#!/usr/bin/env python3
"""Download Polish legal acts from the Sejm ELI API (ISAP) as text records.

Polish legal acts are PUBLIC DOMAIN (Art. 4 of the Polish Copyright Act), so this is
commercial-safe source material. Used both as raw CPT text and as a basis for
synthetic legal Q&A in SFT.

API: https://api.sejm.gov.pl/  (ELI endpoints)
  GET /eli/acts/{publisher}/{year}                 -> list of acts (positions)
  GET /eli/acts/{publisher}/{year}/{pos}           -> act metadata
  GET /eli/acts/{publisher}/{year}/{pos}/text.html -> full text (HTML)
publisher: DU = Dziennik Ustaw, MP = Monitor Polski.

VERIFY at execution: exact endpoint paths, pagination shape, and rate limits — the
API has evolved; adjust field names if the JSON differs.

Usage:
    python scripts/ingest/sejm_isap.py --publisher DU --years 2015-2024 \
        --out data/catalogs/isap/du_2015_2024.jsonl --limit 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.records import JsonlWriter, Record, get_json, get_text, today_iso  # noqa: E402

try:
    import trafilatura
except Exception:  # noqa: BLE001
    trafilatura = None  # text extraction degrades to raw HTML strip

API = "https://api.sejm.gov.pl/eli/acts"


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def extract_text(html: str) -> str:
    if trafilatura is not None:
        out = trafilatura.extract(html, include_comments=False, include_tables=True)
        if out:
            return out
    # crude fallback
    import re
    return re.sub(r"<[^>]+>", " ", html)


def list_acts(session: requests.Session, publisher: str, year: int) -> list[dict]:
    data = get_json(session, f"{API}/{publisher}/{year}")
    # VERIFY: the list of positions is under "items" in the current API.
    return data.get("items", data if isinstance(data, list) else [])


def fetch_act(session: requests.Session, publisher: str, year: int, pos: int) -> Record | None:
    base = f"{API}/{publisher}/{year}/{pos}"
    try:
        meta = get_json(session, base)
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {publisher}/{year}/{pos}: meta fetch failed ({exc})")
        return None

    title = meta.get("title", "")
    # Prefer HTML text; some acts only have PDF (skip PDFs here — handle in processing).
    try:
        html = get_text(session, f"{base}/text.html")
    except Exception:  # noqa: BLE001
        print(f"  no HTML text for {publisher}/{year}/{pos} (PDF-only?) — skipping")
        return None

    text = extract_text(html).strip()
    if len(text) < 200:
        return None

    return Record(
        id=f"isap:{publisher}/{year}/{pos}",
        source="isap",
        url=f"https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id={publisher}{year}{pos:04d}",
        license="public-domain",
        snapshot_date=today_iso(),
        title=title,
        text=text,
        lang="pl",
        meta={"publisher": publisher, "year": year, "pos": pos,
              "type": meta.get("type"), "status": meta.get("status")},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publisher", default="DU", choices=["DU", "MP"])
    ap.add_argument("--years", default="2020-2024", help="e.g. 2024 or 2015-2024")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (per run)")
    args = ap.parse_args()

    session = requests.Session()
    n = 0
    with JsonlWriter(args.out) as w:
        for year in parse_years(args.years):
            acts = list_acts(session, args.publisher, year)
            print(f"[{args.publisher} {year}] {len(acts)} acts listed")
            for act in acts:
                pos = act.get("pos") or act.get("position")
                if pos is None:
                    continue
                rec = fetch_act(session, args.publisher, year, int(pos))
                if rec is not None:
                    w.write(rec)
                    n += 1
                if args.limit and n >= args.limit:
                    print(f"hit --limit {args.limit}")
                    return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
