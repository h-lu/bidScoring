"""Domain models and rules for the evidence-first pipeline."""

from .locator import EvidenceLocator, ImageRegionLocator, TextBBoxLocator
from .models import CitationAssessment, EvidenceWarning
from .verification import CitationVerifier

__all__ = [
    "CitationAssessment",
    "CitationVerifier",
    "EvidenceWarning",
    "EvidenceLocator",
    "TextBBoxLocator",
    "ImageRegionLocator",
]
