import pandas as pd


def deterministic_rerank(scored_candidates: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Placeholder reranker for hackathon demo mode without paid LLM access."""
    return scored_candidates.sort_values("fit_score", ascending=False).head(top_k).reset_index(drop=True)


def build_llm_judge_prompt(jd_text: str, candidates: pd.DataFrame) -> str:
    candidate_blocks = []
    for _, row in candidates.iterrows():
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate ID: {row.get('candidate_id', '')}",
                    f"Name: {row.get('name', '')}",
                    f"Current Title: {row.get('current_title', '')}",
                    f"Fit Score: {row.get('fit_score', '')}",
                    f"Matched Skills: {', '.join(row.get('matched_skills', []))}",
                    f"Missing Must-Have: {', '.join(row.get('missing_must_have', []))}",
                    f"Evidence Profile:\n{row.get('rich_profile', '')}",
                ]
            )
        )

    return f"""You are an expert recruiter and technical hiring evaluator.

Rank the candidates for the job description below. Use only the provided evidence.
Do not infer skills, experience, education, or achievements that are not stated.

Evaluate:
- Must-have coverage
- Skill depth and project evidence
- Experience fit
- Domain and tooling relevance
- Risks or missing requirements

Return JSON with rank, candidate_id, fit_score, strengths, gaps, and reason.

Job Description:
{jd_text}

Candidates:
{chr(10).join(candidate_blocks)}
"""
