from pathlib import Path

from src.data_loader import load_candidates
from src.jd_parser import parse_job_description
from src.profile_builder import build_candidate_profiles
from src.reranker import deterministic_rerank
from src.retrieval import HybridRetriever
from src.scoring import score_candidates


DEFAULT_JD = """We are hiring a Machine Learning Engineer.
Required: Python, NLP, RAG, FAISS, SQL, and 2+ years experience.
Preferred: LangChain, FastAPI, AWS, and recommendation systems."""


def main() -> None:
    candidates = load_candidates(Path("data") / "sample_candidates.csv")
    profiles = build_candidate_profiles(candidates)
    jd_signals = parse_job_description(DEFAULT_JD)

    retriever = HybridRetriever()
    retrieved = retriever.rank(DEFAULT_JD, profiles, top_k=10)
    scored = score_candidates(DEFAULT_JD, jd_signals, retrieved)
    ranked = deterministic_rerank(scored, top_k=10)

    output_path = Path("outputs") / "ranked_candidates.csv"
    ranked.to_csv(output_path, index=False)

    print(ranked[["candidate_id", "name", "current_title", "fit_score", "matched_skills"]].to_string(index=False))
    print(f"\nSaved ranked output to {output_path}")


if __name__ == "__main__":
    main()
