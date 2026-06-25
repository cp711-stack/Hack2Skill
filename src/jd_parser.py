import re

from src.config import COMMON_TECH_SKILLS
from src.preprocess import clean_text


_EXPERIENCE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year)\b", re.IGNORECASE),
    re.compile(r"minimum\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
]


def parse_job_description(jd_text: str) -> dict:
    text = clean_text(jd_text)
    lowered = text.lower()

    matched_skills = sorted(
        skill for skill in COMMON_TECH_SKILLS if _contains_skill(lowered, skill)
    )

    must_have = _extract_marked_skills(lowered, ["must have", "required", "mandatory"])
    nice_to_have = _extract_marked_skills(lowered, ["nice to have", "preferred", "bonus"])

    if not must_have:
        must_have = matched_skills[:]

    return {
        "raw_text": text,
        "skills": matched_skills,
        "must_have_skills": sorted(set(must_have)),
        "nice_to_have_skills": sorted(set(nice_to_have) - set(must_have)),
        "min_experience": _extract_min_experience(text),
    }


def _contains_skill(text: str, skill: str) -> bool:
    pattern = rf"(?<![a-z0-9+#.]){re.escape(skill)}(?![a-z0-9+#.])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _extract_marked_skills(text: str, markers: list[str]) -> list[str]:
    found: list[str] = []
    for marker in markers:
        marker_index = text.find(marker)
        if marker_index == -1:
            continue
        window = text[marker_index : marker_index + 240]
        found.extend(skill for skill in COMMON_TECH_SKILLS if _contains_skill(window, skill))
    return found


def _extract_min_experience(text: str) -> float:
    for pattern in _EXPERIENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return 0.0
