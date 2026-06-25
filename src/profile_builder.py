import pandas as pd

from src.preprocess import clean_text, split_list_field, unique_preserve_order


def build_candidate_profiles(candidates: pd.DataFrame) -> pd.DataFrame:
    profiles = candidates.copy()
    profiles["skill_list"] = profiles["skills"].map(split_list_field)
    profiles["skill_set"] = profiles["skill_list"].map(lambda skills: {skill.lower() for skill in skills})
    profiles["rich_profile"] = profiles.apply(_build_profile_text, axis=1)
    profiles["project_text"] = profiles["projects"].map(clean_text)
    profiles["experience_text"] = profiles["experience"].map(clean_text)
    return profiles


def _build_profile_text(row: pd.Series) -> str:
    skills = unique_preserve_order(split_list_field(row.get("skills", "")))
    certifications = unique_preserve_order(split_list_field(row.get("certifications", "")))

    sections = [
        ("CANDIDATE_ID", row.get("candidate_id", "")),
        ("CURRENT_TITLE", row.get("current_title", "")),
        ("CORE_SKILLS", "; ".join(skills)),
        ("EXPERIENCE_YEARS", row.get("experience_years", "")),
        ("EXPERIENCE_SUMMARY", row.get("experience", "")),
        ("PROJECTS", row.get("projects", "")),
        ("EDUCATION", row.get("education", "")),
        ("CERTIFICATIONS", "; ".join(certifications)),
        ("ACTIVITY_SIGNALS", f"activity_score={row.get('activity_score', 0)}"),
        ("LOCATION", row.get("location", "")),
    ]

    profile_lines = []
    for label, value in sections:
        text = clean_text(value)
        if text:
            profile_lines.append(f"[{label}]\n{text}")
    return "\n\n".join(profile_lines)
