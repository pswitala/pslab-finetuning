"""Holdout routing determinism + DPO pair validation."""

from process.build_dpo import as_text, valid_pair
from process.make_holdout import in_holdout


# --- make_holdout ------------------------------------------------------------

def test_holdout_is_deterministic():
    a = in_holdout("isap:DU/2020/1", 0.5, seed=7)
    b = in_holdout("isap:DU/2020/1", 0.5, seed=7)
    assert a == b


def test_holdout_fraction_roughly_right():
    ids = [f"rec:{i}" for i in range(5000)]
    frac = sum(in_holdout(i, 0.1, seed=1) for i in ids) / len(ids)
    assert 0.07 < frac < 0.13   # ~10% within sampling noise


def test_holdout_seed_changes_split():
    ids = [f"rec:{i}" for i in range(2000)]
    s1 = {i for i in ids if in_holdout(i, 0.1, seed=1)}
    s2 = {i for i in ids if in_holdout(i, 0.1, seed=2)}
    assert s1 != s2


# --- build_dpo ---------------------------------------------------------------

def test_as_text_variants():
    assert as_text("  hi ") == "hi"
    assert as_text([{"role": "user", "content": "a"}, {"role": "x", "content": "b"}]) == "a b"
    assert as_text({"content": "c"}) == "c"


def test_valid_pair_accepts_good():
    assert valid_pair({"prompt": "q", "chosen": "dobra odpowiedź", "rejected": "zła"}, 2)


def test_valid_pair_rejects_bad():
    assert not valid_pair({"prompt": "q", "chosen": "same", "rejected": "same"}, 2)
    assert not valid_pair({"prompt": "", "chosen": "x", "rejected": "y"}, 2)
    assert not valid_pair({"prompt": "q", "chosen": "y"}, 2)  # missing rejected
