import pytest

from scripts.apply_migrations import validate_embedding_dim


def test_validate_embedding_dim_accepts_1536():
    assert validate_embedding_dim(1536) == 1536


def test_validate_embedding_dim_rejects_missing():
    with pytest.raises(ValueError):
        validate_embedding_dim(None)


def test_validate_embedding_dim_rejects_non_1536():
    with pytest.raises(ValueError):
        validate_embedding_dim(3072)
