"""Retrieval configuration loading and keyword expansion helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import yaml

# Type alias for field keywords dictionary
FieldKeywordsDict = Dict[str, List[str]]
SynonymIndexDict = Dict[str, str]  # synonym -> key mapping for bidirectional lookup

logger = logging.getLogger(__name__)

# Default configuration file path (repo_root/config/retrieval_config.yaml)
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "retrieval_config.yaml"
)


def load_retrieval_config(config_path: str | Path | None = None) -> dict:
    """Load retrieval configuration from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("Config file not found: %s. Using empty configuration.", path)
        return {"stopwords": [], "field_keywords": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        config.setdefault("stopwords", [])
        config.setdefault("field_keywords", {})

        logger.debug("Loaded retrieval config from %s", path)
        return config
    except yaml.YAMLError as e:
        logger.error("Failed to parse config file %s: %s", path, e)
        return {"stopwords": [], "field_keywords": {}}
    except Exception as e:  # pragma: no cover
        logger.error("Failed to load config file %s: %s", path, e)
        return {"stopwords": [], "field_keywords": {}}


def build_synonym_index(field_keywords: FieldKeywordsDict) -> SynonymIndexDict:
    """Build a bidirectional synonym index for fast lookup."""
    synonym_index: SynonymIndexDict = {}
    for key, synonyms in field_keywords.items():
        synonym_index[key] = key
        for synonym in synonyms:
            if synonym not in synonym_index:
                synonym_index[synonym] = key
            else:
                existing_key = synonym_index[synonym]
                if existing_key != key:
                    logger.debug(
                        "Synonym '%s' maps to both '%s' and '%s', using first mapping",
                        synonym,
                        existing_key,
                        key,
                    )
    return synonym_index
