from pathlib import Path
from typing import IO

import pandas as pd

from src.preprocess import clean_text, safe_float


REQUIRED_COLUMNS = {
    "candidate_id",
    "name",
    "skills",
    "experience_years",
    "experience",
    "projects",
    "education",
    "activity_score",
}


def load_candidates(source: str | Path | IO) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        data_path = Path(source)
        if not data_path.exists():
            raise FileNotFoundError(f"Candidate data file not found: {data_path}")
        candidates = pd.read_csv(data_path)
    else:
        if hasattr(source, "seek"):
            source.seek(0)
        candidates = pd.read_csv(source)

    missing = REQUIRED_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate data is missing columns: {sorted(missing)}")

    candidates = candidates.copy()
    text_columns = candidates.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        candidates[column] = candidates[column].map(clean_text)

    candidates["experience_years"] = candidates["experience_years"].map(safe_float)
    candidates["activity_score"] = candidates["activity_score"].map(safe_float).clip(0, 1)
    return candidates
