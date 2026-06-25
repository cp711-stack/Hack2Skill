import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9.+#-]*")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_list_field(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    parts = re.split(r"[;,|]", text)
    return [clean_text(part) for part in parts if clean_text(part)]


def normalize_skill(value: str) -> str:
    return clean_text(value).lower()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(clean_text(text))]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = clean_text(value)
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique_values.append(normalized)
    return unique_values
