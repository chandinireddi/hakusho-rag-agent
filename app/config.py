from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root, so the app works no matter which directory it is started from.
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM / embeddings -------------------------------------------------
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.0  # deterministic output → reproducible evaluation

    # --- Paths ------------------------------------------------------------
    documents_dir: Path = ROOT / "documents"
    cache_dir: Path = ROOT / ".cache"

    # --- Chunking (characters; Japanese ≈ 1 char per token) ---------------
    chunk_size: int = Field(500, ge=100, le=2000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    min_chunk_chars: int = 30

    # --- Retrieval --------------------------------------------------------
    retrieve_k: int = Field(8, ge=1, le=30)  # candidates returned by hybrid search
    context_k: int = Field(5, ge=1, le=10)  # chunks actually shown to the LLM
    rrf_k: int = 60  # standard constant from Cormack et al. (2009)

    # --- Agent ------------------------------------------------------------
    max_query_rewrites: int = Field(1, ge=0, le=3)
    max_history_turns: int = Field(3, ge=0, le=10)  # past Q&A pairs used to resolve follow-up questions

    # --- Observability (optional) -----------------------------------------
    # Set LANGSMITH_API_KEY to trace every graph node and LLM call in LangSmith.
    langsmith_api_key: str = ""
    langsmith_project: str = "hakusho-research"

    # --- API --------------------------------------------------------------
    allowed_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
