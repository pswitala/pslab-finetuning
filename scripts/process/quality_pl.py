#!/usr/bin/env python3
"""Polish-specific quality heuristics for corpus filtering.

Pure-Python, no heavy deps — usable standalone and as datatrove custom filters.
These complement (don't replace) standard Gopher/C4 rules and fastText language-ID.

Run a quick self-test (no GPU/network):
    python scripts/process/quality_pl.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Polish-specific letters. Legitimate Polish prose has a characteristic, non-trivial
# frequency of these; near-zero usually means non-Polish or garbled/OCR text.
PL_DIACRITICS = set("ąęóśżźćńłĄĘÓŚŻŹĆŃŁ")

# Common Polish function words — cheap signal that text is actually Polish.
PL_STOPWORDS = {
    "i", "w", "na", "z", "do", "nie", "że", "to", "się", "jest", "o", "a",
    "od", "po", "za", "jak", "oraz", "lub", "dla", "przez", "ten", "który",
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class QualityResult:
    keep: bool
    reason: str = ""
    diacritic_ratio: float = 0.0
    stopword_ratio: float = 0.0
    mean_word_len: float = 0.0
    n_words: int = 0


def diacritic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in PL_DIACRITICS for c in letters) / len(letters)


def stopword_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(w.lower() in PL_STOPWORDS for w in words) / len(words)


def assess(
    text: str,
    *,
    min_words: int = 50,
    max_words: int = 100_000,
    min_diacritic_ratio: float = 0.008,   # ~0.8% of letters; Polish prose is well above
    min_stopword_ratio: float = 0.02,
    min_mean_word_len: float = 2.5,
    max_mean_word_len: float = 12.0,
    min_alpha_ratio: float = 0.6,
) -> QualityResult:
    """Apply Polish-tuned Gopher-style heuristics. Returns keep/reject + diagnostics."""
    words = _WORD_RE.findall(text)
    n = len(words)
    if n < min_words:
        return QualityResult(False, f"too_few_words({n})", n_words=n)
    if n > max_words:
        return QualityResult(False, f"too_many_words({n})", n_words=n)

    alpha = sum(c.isalpha() for c in text)
    alpha_ratio = alpha / max(len(text), 1)
    if alpha_ratio < min_alpha_ratio:
        return QualityResult(False, f"low_alpha_ratio({alpha_ratio:.2f})", n_words=n)

    mwl = sum(len(w) for w in words) / n
    if not (min_mean_word_len <= mwl <= max_mean_word_len):
        return QualityResult(False, f"mean_word_len({mwl:.1f})",
                             mean_word_len=mwl, n_words=n)

    dr = diacritic_ratio(text)
    if dr < min_diacritic_ratio:
        return QualityResult(False, f"low_diacritics({dr:.4f})",
                             diacritic_ratio=dr, n_words=n)

    sr = stopword_ratio(words)
    if sr < min_stopword_ratio:
        return QualityResult(False, f"low_stopwords({sr:.3f})",
                             stopword_ratio=sr, n_words=n)

    return QualityResult(True, "ok", dr, sr, mwl, n)


def _selftest() -> None:
    good = (
        "Rzeczpospolita Polska jest demokratycznym państwem prawnym. "
        "W ustawie zasadniczej określono podstawowe prawa i wolności obywateli, "
        "które są chronione przez niezależne sądy oraz inne organy władzy publicznej. "
        "Każdy ma prawo do rzetelnego rozpatrzenia jego sprawy w rozsądnym terminie. " * 4
    )
    english = (
        "The quick brown fox jumps over the lazy dog. This is an English paragraph "
        "with no Polish diacritics whatsoever and should be rejected by the filter. " * 5
    )
    garbage = "aaa bbb ccc " * 60

    r_good = assess(good)
    r_en = assess(english)
    r_garbage = assess(garbage)
    print("good   :", r_good)
    print("english:", r_en)
    print("garbage:", r_garbage)
    assert r_good.keep, "expected Polish text to pass"
    assert not r_en.keep, "expected English text to fail (low diacritics)"
    assert not r_garbage.keep, "expected garbage to fail"
    print("\nquality_pl self-test OK")


if __name__ == "__main__":
    _selftest()
