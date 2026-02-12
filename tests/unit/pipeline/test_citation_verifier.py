from bid_scoring.pipeline.domain.verification import CitationVerifier


def test_citation_verifier_returns_verified_when_hash_and_anchor_ok():
    verifier = CitationVerifier()

    result = verifier.assess(
        {
            "ok": True,
            "citation_id": "c-1",
            "unit_id": "u-1",
            "hash_ok": True,
            "anchor_ok": True,
        }
    )

    assert result.status == "verified"
    assert result.evidence_status == "verified"
    assert result.warnings == []


def test_citation_verifier_returns_warning_for_hash_mismatch():
    verifier = CitationVerifier()

    result = verifier.assess(
        {
            "ok": False,
            "citation_id": "c-2",
            "unit_id": "u-2",
            "hash_ok": False,
            "anchor_ok": True,
        }
    )

    assert result.status == "warning"
    assert result.evidence_status == "unverifiable"
    assert [w.code for w in result.warnings] == ["hash_mismatch"]


def test_citation_verifier_returns_warning_for_not_found():
    verifier = CitationVerifier()

    result = verifier.assess(
        {
            "ok": False,
            "citation_id": "missing",
            "reason": "not_found",
        }
    )

    assert result.status == "warning"
    assert result.evidence_status == "unverifiable"
    assert [w.code for w in result.warnings] == ["citation_not_found"]
