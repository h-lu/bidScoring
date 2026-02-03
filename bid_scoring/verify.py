# bid_scoring/verify.py
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Remove all whitespace for flexible matching
    text = re.sub(r"\s+", "", text)
    return text.lower()


def verify_citation(cited_text: str, original_text: str):
    cited_clean = normalize(cited_text)
    original_clean = normalize(original_text)
    if not cited_clean:
        return {"verified": False, "match_type": "no_match"}
    if cited_clean in original_clean:
        return {"verified": True, "match_type": "exact_normalized"}
    return {"verified": False, "match_type": "no_match"}
