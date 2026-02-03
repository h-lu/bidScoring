# tests/test_verify_citations.py
from bid_scoring.verify import verify_citation


def test_verify_exact_match():
    res = verify_citation("培训时间：2天", "培训时间：2天，含安装培训")
    assert res["verified"] is True
