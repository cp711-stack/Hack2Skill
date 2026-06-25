import pandas as pd

from src.config import DEFAULT_WEIGHTS, RankingWeights
from src.preprocess import normalize_skill, safe_float


def score_candidates(
    jd_text: str,
    jd_signals: dict,
    retrieved_candidates: pd.DataFrame,
    weights: RankingWeights = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    scored = retrieved_candidates.copy()
    if scored.empty:
        return scored

    must_have = set(map(normalize_skill, jd_signals.get("must_have_skills", [])))
    nice_to_have = set(map(normalize_skill, jd_signals.get("nice_to_have_skills", [])))
    all_jd_skills = set(map(normalize_skill, jd_signals.get("skills", [])))
    min_experience = safe_float(jd_signals.get("min_experience", 0))

    scored["must_have_score"] = scored["skill_set"].map(lambda skills: _coverage(skills, must_have))
    scored["nice_to_have_score"] = scored["skill_set"].map(lambda skills: _coverage(skills, nice_to_have))
    scored["skill_overlap_score"] = scored["skill_set"].map(lambda skills: _coverage(skills, all_jd_skills))
    scored["experience_score"] = scored["experience_years"].map(
        lambda years: _experience_fit(safe_float(years), min_experience)
    )
    scored["project_score"] = scored.apply(
        lambda row: _project_relevance(row.get("project_text", ""), all_jd_skills),
        axis=1,
    )
    scored["activity_score"] = scored["activity_score"].map(lambda value: min(max(safe_float(value), 0), 1))

    scored["fit_score"] = 100 * (
        weights.semantic * scored["semantic_score"]
        + weights.bm25 * scored["bm25_score"]
        + weights.must_have * scored["must_have_score"]
        + weights.nice_to_have * scored["nice_to_have_score"]
        + weights.experience * scored["experience_score"]
        + weights.project * scored["project_score"]
        + weights.activity * scored["activity_score"]
    )
    scored["fit_score"] = scored["fit_score"].round(2)

    scored["matched_skills"] = scored["skill_set"].map(
        lambda skills: sorted(skills & all_jd_skills)
    )
    scored["missing_must_have"] = scored["skill_set"].map(
        lambda skills: sorted(must_have - skills)
    )
    scored["strengths"] = scored.apply(_strengths, axis=1)
    scored["gaps"] = scored.apply(_gaps, axis=1)
    scored["reason"] = scored.apply(_reason, axis=1)

    return scored.sort_values("fit_score", ascending=False).reset_index(drop=True)


def _coverage(candidate_skills: set[str], required_skills: set[str]) -> float:
    if not required_skills:
        return 1.0
    return len(candidate_skills & required_skills) / len(required_skills)


def _experience_fit(years: float, min_years: float) -> float:
    if min_years <= 0:
        return 1.0
    if years >= min_years:
        return 1.0
    return max(0.0, years / min_years)


def _project_relevance(project_text: str, jd_skills: set[str]) -> float:
    if not jd_skills:
        return 0.5
    lowered = str(project_text).lower()
    mentions = sum(1 for skill in jd_skills if skill in lowered)
    return min(1.0, mentions / max(1, min(len(jd_skills), 4)))


def _strengths(row: pd.Series) -> list[str]:
    strengths: list[str] = []
    if row["must_have_score"] >= 0.8:
        strengths.append("Strong must-have skill coverage")
    if row["semantic_score"] >= 0.7:
        strengths.append("High semantic alignment with the JD")
    if row["experience_score"] >= 1:
        strengths.append("Meets or exceeds experience expectation")
    if row["project_score"] >= 0.5:
        strengths.append("Relevant project evidence")
    if row["activity_score"] >= 0.75:
        strengths.append("Strong activity or engagement signal")
    return strengths or ["Potential fit based on available profile evidence"]


def _gaps(row: pd.Series) -> list[str]:
    gaps: list[str] = []
    missing = row.get("missing_must_have", [])
    if missing:
        gaps.append(f"Missing explicit must-have skills: {', '.join(missing)}")
    if row["experience_score"] < 1:
        gaps.append("Experience may be below the stated requirement")
    if row["project_score"] < 0.25:
        gaps.append("Limited direct project evidence for this JD")
    return gaps or ["No major gaps found from structured data"]


def _reason(row: pd.Series) -> str:
    matched = ", ".join(row.get("matched_skills", [])[:6]) or "general profile signals"
    return (
        f"{row.get('name', 'Candidate')} ranks well due to {matched}, "
        f"with a fit score of {row['fit_score']}."
    )
