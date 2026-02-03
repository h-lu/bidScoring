# tests/test_citation_validation.py
"""Tests for citation validation functionality."""

import pytest
from bid_scoring.verify import verify_citation, normalize


class TestNormalize:
    """Test text normalization function."""

    def test_normalize_unicode(self):
        """Should normalize unicode characters."""
        text = "培训时间：２天"  # Full-width numbers
        result = normalize(text)
        assert "2" in result  # Should be normalized to half-width

    def test_normalize_whitespace(self):
        """Should normalize whitespace."""
        text = "培训  时间\n\n2天"
        result = normalize(text)
        assert "  " not in result
        assert "\n" not in result

    def test_normalize_strips_edges(self):
        """Should strip whitespace from edges."""
        text = "  培训时间  "
        result = normalize(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestVerifyCitation:
    """Test citation verification."""

    def test_exact_match(self):
        """Should verify exact citation match."""
        res = verify_citation("培训时间：2天", "培训时间：2天，含安装培训")
        assert res["verified"] is True
        assert res["match_type"] == "exact_normalized"

    def test_normalized_match(self):
        """Should match after normalization."""
        res = verify_citation("培训时间：2天", "培训  时间：２天，含安装培训")
        assert res["verified"] is True

    def test_no_match(self):
        """Should return not verified when citation not found."""
        res = verify_citation("不存在的文本", "培训时间：2天")
        assert res["verified"] is False
        assert res["match_type"] == "no_match"

    def test_empty_citation(self):
        """Should not verify empty citation."""
        res = verify_citation("", "培训时间：2天")
        assert res["verified"] is False

    def test_partial_match_not_verified(self):
        """Should not verify partial matches."""
        res = verify_citation("培训时间：3天", "培训时间：2天，含安装培训")
        assert res["verified"] is False

    def test_case_insensitive_match(self):
        """Should match case-insensitively for ASCII."""
        res = verify_citation("Training", "training schedule")
        assert res["verified"] is True
