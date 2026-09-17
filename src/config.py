from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL_ALIASES: dict[str, str] = {
    "gpt-120": "openai/gpt-oss-120b",
    "gpt 120": "openai/gpt-oss-120b",
    "gpt120": "openai/gpt-oss-120b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "120b": "openai/gpt-oss-120b",
    "llama3": "llama-3.3-70b-versatile",
    "llama-3.3": "llama-3.3-70b-versatile",
    "llama-3": "llama-3.3-70b-versatile",
}


def normalize_model_name(name: str) -> str:
    cleaned = name.strip().lower()
    return MODEL_ALIASES.get(cleaned, name.strip())


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    policy_dir: Path = data_dir / "policies"
    upload_dir: Path = data_dir / "uploads"
    chroma_dir: Path = project_root / "chroma_db"
    collection_name: str = os.getenv("CHROMA_COLLECTION", "enterprise-policy-assistant")

    # LLM Settings (Groq)
    llm_provider: str = "groq"
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = normalize_model_name(os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

    # Retrieval & Chunking
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Security & Guardrails
    enable_llm_injection_check: bool = os.getenv("ENABLE_LLM_INJECTION_CHECK", "true").lower() == "true"
    enable_groundedness_check: bool = os.getenv("ENABLE_GROUNDEDNESS_CHECK", "true").lower() == "true"
    groundedness_threshold: float = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.75"))

    @property
    def active_provider(self) -> str:
        return "groq"

    @property
    def active_model(self) -> str:
        return self.groq_model

    @property
    def all_document_dirs(self) -> list[Path]:
        return [self.policy_dir]


settings = Settings()
settings.policy_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
