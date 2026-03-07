"""Tests for fingerprint-based cache invalidation."""
from harnessml.core.models.fingerprint import Fingerprint


def test_fingerprint_match(tmp_path):
    fp = Fingerprint.compute({"type": "xgb", "depth": 3}, "hash123", 1000.0)
    fp.save(tmp_path / ".fingerprint")
    assert fp.matches(tmp_path / ".fingerprint")


def test_fingerprint_mismatch_on_config_change(tmp_path):
    fp1 = Fingerprint.compute({"type": "xgb", "depth": 3}, "hash123", 1000.0)
    fp1.save(tmp_path / ".fingerprint")
    fp2 = Fingerprint.compute({"type": "xgb", "depth": 5}, "hash123", 1000.0)
    assert not fp2.matches(tmp_path / ".fingerprint")


def test_fingerprint_mismatch_on_data_hash_change(tmp_path):
    fp1 = Fingerprint.compute({"type": "xgb"}, "hash_v1", 1000.0)
    fp1.save(tmp_path / ".fingerprint")
    fp2 = Fingerprint.compute({"type": "xgb"}, "hash_v2", 1000.0)
    assert not fp2.matches(tmp_path / ".fingerprint")


def test_fingerprint_mismatch_on_data_size_change(tmp_path):
    fp1 = Fingerprint.compute({"type": "xgb"}, "hash123", 1000.0)
    fp1.save(tmp_path / ".fingerprint")
    fp2 = Fingerprint.compute({"type": "xgb"}, "hash123", 2000.0)
    assert not fp2.matches(tmp_path / ".fingerprint")


def test_fingerprint_missing_file(tmp_path):
    fp = Fingerprint.compute({"type": "xgb"}, "hash123", 1000.0)
    assert not fp.matches(tmp_path / "nonexistent")


def test_fingerprint_deterministic():
    """Same inputs always produce the same config_hash."""
    fp1 = Fingerprint.compute({"b": 2, "a": 1}, "h", 10.0)
    fp2 = Fingerprint.compute({"a": 1, "b": 2}, "h", 10.0)
    assert fp1.config_hash == fp2.config_hash


def test_fingerprint_corrupted_file(tmp_path):
    """Corrupted JSON should return False, not crash."""
    bad_file = tmp_path / ".fingerprint"
    bad_file.write_text("not json at all")
    fp = Fingerprint.compute({"type": "xgb"}, "hash123", 1000.0)
    assert not fp.matches(bad_file)
