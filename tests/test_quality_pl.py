"""Polish quality heuristics (scripts/process/quality_pl.py)."""

from process.quality_pl import assess, diacritic_ratio, stopword_ratio

_POLISH = (
    "W tym roku wiele osób nie wie, że nowe przepisy prawne wprowadzają zmiany, "
    "które mają istotny wpływ na życie codzienne obywateli w całym kraju. "
) * 5  # ~90+ words, rich in Polish stopwords and diacritics


def test_keeps_good_polish():
    res = assess(_POLISH)
    assert res.keep, res.reason


def test_rejects_too_short():
    res = assess("krótki tekst bez treści")
    assert not res.keep


def test_diacritic_ratio_detects_polish():
    assert diacritic_ratio("zażółć gęślą jaźń") > 0.0
    assert diacritic_ratio("plain english text") == 0.0


def test_stopword_ratio():
    assert stopword_ratio("to jest i w na z") > 0.0
