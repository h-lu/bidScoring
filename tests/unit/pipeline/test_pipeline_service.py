from bid_scoring.pipeline.application.service import PipelineService


class _FakeVerifier:
    def assess(self, verification):  # pragma: no cover - exercised via service flow
        if verification.get("ok"):
            return type(
                "Result",
                (),
                {"status": "verified", "evidence_status": "verified", "warnings": []},
            )()
        return type(
            "Result",
            (),
            {
                "status": "warning",
                "evidence_status": "unverifiable",
                "warnings": [
                    type(
                        "Warn",
                        (),
                        {"code": "hash_mismatch", "message": "hash mismatch"},
                    )()
                ],
            },
        )()


def test_pipeline_service_keeps_completed_status_with_unverifiable_warnings():
    checks = {
        "c-1": {"ok": True, "citation_id": "c-1", "hash_ok": True, "anchor_ok": True},
        "c-2": {"ok": False, "citation_id": "c-2", "hash_ok": False, "anchor_ok": True},
    }

    def fake_verify(_conn, *, citation_id):
        return checks[citation_id]

    service = PipelineService(verify_citation_fn=fake_verify, verifier=_FakeVerifier())
    summary = service.evaluate_citations(conn=object(), citation_ids=["c-1", "c-2"])

    assert summary.status == "completed"
    assert summary.total_citations == 2
    assert summary.verified_citations == 1
    assert summary.unverifiable_citations == 1
    assert summary.warning_codes == ["hash_mismatch"]
