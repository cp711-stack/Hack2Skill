from io import StringIO

from src.data_loader import load_candidates
from src.jd_parser import parse_job_description
from src.profile_builder import build_candidate_profiles
from src.retrieval import HybridRetriever
from src.scoring import score_candidates


def test_pipeline_returns_ranked_candidates():
    candidates = load_candidates("data/sample_candidates.csv")
    profiles = build_candidate_profiles(candidates)
    jd = "We need a Python NLP engineer with RAG, FAISS, SQL, and 2+ years experience."
    jd_signals = parse_job_description(jd)

    retriever = HybridRetriever()
    retrieval_results = retriever.rank(jd, profiles, top_k=3)
    scored = score_candidates(jd, jd_signals, retrieval_results)

    assert len(scored) == 3
    assert scored.iloc[0]["fit_score"] >= scored.iloc[-1]["fit_score"]
    assert "candidate_id" in scored.columns


def test_loader_accepts_file_like_csv():
    csv_data = StringIO(
        "candidate_id,name,skills,experience_years,experience,projects,education,activity_score\n"
        "C100,Test Candidate,Python; SQL,2,Data work,Analytics project,B.Tech,0.7\n"
    )

    candidates = load_candidates(csv_data)

    assert candidates.iloc[0]["candidate_id"] == "C100"
    assert candidates.iloc[0]["experience_years"] == 2
