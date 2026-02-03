# tests/test_eval_metrics.py
import os
import pytest

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set", allow_module_level=True)

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test


def test_eval_smoke():
    case = LLMTestCase(
        input="培训方案是否完整？",
        actual_output="培训包含2天安装与操作培训。",
        expected_output="包含培训时长与内容。",
        retrieval_context=["培训时间：2天，含安装培训、操作培训"]
    )
    assert_test(case, [AnswerRelevancyMetric(threshold=0.5)])
