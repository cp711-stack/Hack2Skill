from dataclasses import dataclass


@dataclass(frozen=True)
class RankingWeights:
    semantic: float = 0.40
    bm25: float = 0.12
    must_have: float = 0.20
    nice_to_have: float = 0.08
    experience: float = 0.10
    project: float = 0.06
    activity: float = 0.04


DEFAULT_WEIGHTS = RankingWeights()

DEFAULT_TOP_K = 10

COMMON_TECH_SKILLS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "nosql",
    "mongodb",
    "postgresql",
    "mysql",
    "react",
    "node",
    "node.js",
    "django",
    "flask",
    "fastapi",
    "streamlit",
    "machine learning",
    "deep learning",
    "nlp",
    "rag",
    "llm",
    "langchain",
    "transformers",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "pandas",
    "numpy",
    "spark",
    "airflow",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "faiss",
    "chromadb",
    "vector database",
    "elasticsearch",
    "recommendation system",
    "data analysis",
    "data visualization",
    "power bi",
    "tableau",
    "excel",
    "git",
    "rest api",
    "graphql",
}
