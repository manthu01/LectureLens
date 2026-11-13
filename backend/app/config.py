from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.8-flash"
    gemini_api_key: str = ""

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # "translate" outputs English whatever the spoken language (needed for Hindi/Hinglish
    # lectures, since embeddings are English-only). "transcribe" keeps the original language.
    whisper_task: str = "translate"

    # Browser name (chrome, edge, firefox...) whose cookies yt-dlp may use if YouTube blocks anonymous downloads.
    ytdlp_cookies_from_browser: str = ""

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    database_url: str = ""

    data_dir: str = "data"
    cors_origins: str = "http://localhost:3000"
    top_k: int = 6
    retrieval_mode: str = "hybrid"  # dense | bm25 | hybrid
    # Cross-encoder that scores the top_k hits (order unchanged) for the refusal gate. Empty = off.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Refuse without calling the LLM when the best rerank score (0..1) is below this. 0 disables the gate.
    rerank_threshold: float = 0.01
    # Public demo mode: adding/retrying lectures is disabled; search, answers, summaries, quizzes still work.
    read_only: bool = False
    # Max LLM-backed requests (ask, quiz) per client IP per minute. 0 = unlimited.
    rate_limit_per_min: int = 0
    classify_segments: bool = True  # label segments teaching/off_topic/admin/exam_hint at ingest

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else ROOT / p

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_path / 'lecturelens.db').as_posix()}"


settings = Settings()
