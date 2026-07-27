"""MinHash-LSH (num_buckets, hashes_per_bucket) band split derived from the threshold."""

from process.dedup import _lsh_params_for_threshold


def _approx_threshold(num_buckets: int, hashes_per_bucket: int) -> float:
    return (1.0 / num_buckets) ** (1.0 / hashes_per_bucket)


def test_permutation_budget_is_respected():
    # Total permutations (b*r) must stay within the requested budget and be sane
    # (the old heuristic produced only ~48 perms for t=0.8).
    b, r = _lsh_params_for_threshold(0.8, num_permutations=112)
    assert b * r <= 112
    assert b * r >= 96  # uses most of the budget, not a tiny fraction


def test_approximates_target_threshold():
    for target in (0.6, 0.8, 0.9):
        b, r = _lsh_params_for_threshold(target, num_permutations=128)
        assert abs(_approx_threshold(b, r) - target) < 0.05


def test_lower_threshold_more_buckets():
    # Lower similarity threshold -> more bands (more candidate pairs).
    b_low, _ = _lsh_params_for_threshold(0.6, num_permutations=112)
    b_high, _ = _lsh_params_for_threshold(0.9, num_permutations=112)
    assert b_low > b_high


def test_always_valid():
    b, r = _lsh_params_for_threshold(0.99, num_permutations=112)
    assert b >= 1 and r >= 1
