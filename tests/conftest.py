from __future__ import annotations

import uuid

import pytest


@pytest.fixture
def fixed_ids() -> dict[str, str]:
    return {
        "project_id": str(uuid.UUID("11111111-1111-1111-1111-111111111111")),
        "document_id": str(uuid.UUID("22222222-2222-2222-2222-222222222222")),
        "version_id": str(uuid.UUID("33333333-3333-3333-3333-333333333333")),
    }

