"""Holdout id-manifest loading used by the SFT/CPT builders to prevent eval leakage."""

from common.records import load_exclude_ids


def test_empty_path_returns_empty_set():
    assert load_exclude_ids(None) == set()
    assert load_exclude_ids("") == set()


def test_loads_ids_stripping_blank_lines(tmp_path):
    manifest = tmp_path / "ids.txt"
    manifest.write_text("isap:2020:1\n\ngus:K11:42\n  \ndane:xyz\n", encoding="utf-8")
    ids = load_exclude_ids(str(manifest))
    assert ids == {"isap:2020:1", "gus:K11:42", "dane:xyz"}


def test_excluded_record_is_filtered(tmp_path):
    # The builders' filter is `rec.get("id") in exclude_ids`; verify the set membership
    # the builders rely on holds for held-out ids and lets others through.
    manifest = tmp_path / "ids.txt"
    manifest.write_text("held-1\nheld-2\n", encoding="utf-8")
    ids = load_exclude_ids(str(manifest))
    records = [{"id": "held-1"}, {"id": "keep-1"}, {"id": "held-2"}, {"id": "keep-2"}]
    kept = [r for r in records if r.get("id") not in ids]
    assert [r["id"] for r in kept] == ["keep-1", "keep-2"]
