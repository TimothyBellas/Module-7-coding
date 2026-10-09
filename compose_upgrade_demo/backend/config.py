"""Central configuration: process environment > project .env > defaults."""

from pathlib import Path

from pydantic import AnyHttpUrl, Field, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Load and validate settings once when the API process starts.

    Compose supplies .env values through env_file. For a local Python run,
    Pydantic reads the same project-root .env directly, regardless of cwd.
    An invalid value raises a clear validation error instead of being clamped.
    """

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR.parent / ".env",
        env_file_encoding="utf-8-sig",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",  # The shared .env also contains Compose-only port values.
        str_strip_whitespace=True,
    )

    OLLAMA_URL: str = "http://localhost:11434"
    MODEL_NAME: str = Field(default="llama3.2:1b", min_length=1)
    CHROMA_PATH: Path = BACKEND_DIR / "chroma_data"
    MAX_RESULTS: int = Field(default=3, ge=1, le=100)
    CONFIDENCE_THRESHOLD: float = Field(default=1.0, gt=0, le=2, allow_inf_nan=False)
    DEBUG: bool = False

    DOCS_PATH: Path = BACKEND_DIR / "docs"
    COLLECTION_NAME: str = Field(default="documents", min_length=3, max_length=63)
    OLLAMA_TIMEOUT: float = Field(default=180, gt=0, allow_inf_nan=False)
    OLLAMA_CONNECT_TIMEOUT: float = Field(default=10, gt=0, allow_inf_nan=False)
    HEALTH_TIMEOUT: float = Field(default=3, gt=0, allow_inf_nan=False)
    TEMPERATURE: float = Field(default=0.1, ge=0, le=2, allow_inf_nan=False)
    INGEST_BATCH_SIZE: int = Field(default=100, ge=1, le=1000)
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])
    CHROMA_TELEMETRY: bool = False

    @field_validator("OLLAMA_URL")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        """Accept an HTTP(S) URL and normalize the slash used by API requests."""
        return str(TypeAdapter(AnyHttpUrl).validate_python(value)).rstrip("/")

    @field_validator("CHROMA_PATH", "DOCS_PATH")
    @classmethod
    def resolve_local_path(cls, value: Path) -> Path:
        """Relative paths refer to backend/, not the current terminal directory."""
        return value if value.is_absolute() else BACKEND_DIR / value
