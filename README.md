# Intelligent Candidate Discovery

Hackathon proof of concept for ranking candidates against a job description using hybrid retrieval, structured candidate profiles, metadata-aware scoring, and optional LLM reranking.

## What It Does

- Loads candidate profile data from CSV.
- Builds section-aware rich candidate profiles from skills, experience, projects, education, and activity signals.
- Parses a job description into lightweight matching signals.
- Retrieves candidates with a hybrid of semantic similarity and keyword/BM25 matching.
- Scores candidates using configurable weights.
- Shows an explainable ranked shortlist in a Streamlit app.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── sample_candidates.csv
├── outputs/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── jd_parser.py
│   ├── preprocess.py
│   ├── profile_builder.py
│   ├── retrieval.py
│   ├── scoring.py
│   └── reranker.py
└── tests/
    └── test_pipeline.py
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

You can also run the ranking pipeline without the UI:

```bash
python run_ranking.py
```

This writes `outputs/ranked_candidates.csv`.

If you want stronger local embeddings later, install optional dependencies:

```bash
pip install -r requirements-optional.txt
```

If `sentence-transformers` is unavailable or the model cannot be downloaded, the app falls back to TF-IDF semantic matching so the demo still works offline.

## Ranking Method

The system uses a two-stage ranking strategy:

1. **Hybrid retrieval**
   - Semantic similarity from sentence embeddings or TF-IDF fallback.
   - BM25 keyword relevance.
   - Exact skill overlap.

2. **Candidate scoring**
   - Semantic score.
   - Must-have skill coverage.
   - Nice-to-have skill coverage.
   - Experience fit.
   - Project relevance.
   - Activity or engagement signal.

The output includes fit score, matched skills, missing skills, strengths, gaps, and a concise recruiter-facing explanation.

## Next Improvements

- Add support for uploaded resume PDFs.
- Add OpenAI or local LLM reranking for final top candidates.
- Add evaluation metrics against labeled JD-candidate pairs.
- Add FAISS indexing for larger datasets.
- Add fairness guardrails and sensitive attribute filtering.
