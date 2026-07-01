"""MinHash-LSH band count derived from the target Jaccard threshold."""

from process.dedup import _num_buckets_for_threshold


def test_num_buckets_for_threshold():
    # t = (1/b)^(1/r)  =>  b ≈ t^(-r). For t=0.8, r=8 -> ~6.
    assert _num_buckets_for_threshold(0.8, 8) == 6


def test_lower_threshold_more_buckets():
    # Lower similarity threshold -> more bands (more candidate pairs).
    assert (_num_buckets_for_threshold(0.6, 8)
            > _num_buckets_for_threshold(0.9, 8))


def test_always_at_least_one():
    assert _num_buckets_for_threshold(0.99, 8) >= 1
