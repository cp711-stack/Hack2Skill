from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_TOP_K
from src.data_loader import load_candidates
from src.jd_parser import parse_job_description
from src.profile_builder import build_candidate_profiles
from src.reranker import build_llm_judge_prompt, deterministic_rerank
from src.retrieval import HybridRetriever
from src.scoring import score_candidates


SAMPLE_JD = """We are hiring a Machine Learning Engineer to build intelligent candidate discovery systems.
Required: Python, NLP, RAG, FAISS or vector databases, SQL, and 2+ years experience.
Preferred: LangChain, FastAPI, AWS, recommendation systems, and strong project evidence."""


st.set_page_config(
    page_title="Intelligent Candidate Discovery",
    page_icon="",
    layout="wide",
)


def main() -> None:
    st.title("Intelligent Candidate Discovery")

    with st.sidebar:
        st.header("Candidate Data")
        st.caption("Build: upload-fix-v2")
        uploaded_file = st.file_uploader("Upload candidate CSV", type=["csv"])
        top_k = st.slider("Shortlist size", min_value=3, max_value=25, value=DEFAULT_TOP_K)
        show_prompt = st.checkbox("Show optional LLM judge prompt", value=False)

    jd_text = st.text_area("Job description", value=SAMPLE_JD, height=180)

    if st.button("Rank candidates", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("Add a job description before ranking.")
            return

        with st.spinner("Reading candidate profiles..."):
            candidates = _load_data(uploaded_file)
            profiles = build_candidate_profiles(candidates)
            jd_signals = parse_job_description(jd_text)

        with st.spinner("Running hybrid retrieval and scoring..."):
            retriever = HybridRetriever()
            retrieved = retriever.rank(jd_text, profiles, top_k=max(top_k, 10))
            scored = score_candidates(jd_text, jd_signals, retrieved)
            ranked = deterministic_rerank(scored, top_k=top_k)

        _render_summary(jd_signals, retriever.backend.name, ranked)
        _render_leaderboard(ranked)
        _render_candidate_details(ranked)

        if show_prompt:
            st.subheader("Optional LLM Judge Prompt")
            st.code(build_llm_judge_prompt(jd_text, ranked), language="text")


def _load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return load_candidates(uploaded_file)
    return load_candidates(Path("data") / "sample_candidates.csv")


def _render_summary(jd_signals: dict, backend_name: str, ranked: pd.DataFrame) -> None:
    st.subheader("Ranking Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates ranked", len(ranked))
    col2.metric("Minimum experience", f"{jd_signals['min_experience']:.0f}+ yrs")
    col3.metric("JD skills found", len(jd_signals["skills"]))
    col4.metric("Retrieval backend", backend_name)

    st.caption(
        "Must-have skills detected: "
        + (", ".join(jd_signals["must_have_skills"]) or "none")
    )


def _render_leaderboard(ranked: pd.DataFrame) -> None:
    st.subheader("Ranked Shortlist")
    view = ranked[
        [
            "candidate_id",
            "name",
            "current_title",
            "fit_score",
            "semantic_score",
            "must_have_score",
            "experience_score",
            "matched_skills",
            "missing_must_have",
        ]
    ].copy()
    view.insert(0, "rank", range(1, len(view) + 1))
    st.dataframe(view, use_container_width=True, hide_index=True)

    csv = ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranked output",
        data=csv,
        file_name="ranked_candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_candidate_details(ranked: pd.DataFrame) -> None:
    st.subheader("Candidate Evidence")
    for index, row in ranked.iterrows():
        label = f"#{index + 1} {row['name']} - {row['current_title']} - {row['fit_score']}"
        with st.expander(label, expanded=index == 0):
            col1, col2 = st.columns(2)
            col1.markdown("**Strengths**")
            col1.write("\n".join(f"- {item}" for item in row["strengths"]))
            col2.markdown("**Gaps**")
            col2.write("\n".join(f"- {item}" for item in row["gaps"]))
            st.markdown("**Reason**")
            st.write(row["reason"])
            st.markdown("**Rich Candidate Profile**")
            st.code(row["rich_profile"], language="text")


if __name__ == "__main__":
    main()
