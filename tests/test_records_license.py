"""Commercial-safe license logic (centralized in scripts/common/records.py)."""

from common.records import is_commercial_safe, normalize_license


def test_normalize_license():
    assert normalize_license("CC BY 4.0") == "cc-by-4.0"
    assert normalize_license(None) == "unknown"
    assert normalize_license("") == "unknown"


def test_permissive_licenses_are_safe():
    for lic in ["CC0", "cc-by-4.0", "public-domain", "PDDL", "odc-by", "Apache-2.0", "MIT"]:
        assert is_commercial_safe(lic), lic


def test_noncommercial_and_noderiv_rejected():
    # These start with an allowed prefix but are NOT commercial-safe.
    assert not is_commercial_safe("cc-by-nc-4.0")
    assert not is_commercial_safe("cc-by-nd-4.0")
    assert not is_commercial_safe("cc-by-nc-sa-4.0")


def test_unknown_and_proprietary_rejected():
    assert not is_commercial_safe("unknown")
    assert not is_commercial_safe(None)
    assert not is_commercial_safe("proprietary")
