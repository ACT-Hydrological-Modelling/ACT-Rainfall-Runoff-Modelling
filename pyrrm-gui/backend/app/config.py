"""
Application configuration settings.

Uses pydantic-settings for environment variable management.
"""

from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "pyrrm-gui"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./data/pyrrm.db"
    
    # Redis (for Celery and WebSocket)
    redis_url: str = "redis://localhost:6379/0"
    
    # File Storage
    data_dir: Path = Path("./data")
    uploads_dir: Path = Path("./data/uploads")
    reports_dir: Path = Path("./data/reports")
    checkpoints_dir: Path = Path("./data/checkpoints")
    
    # Calibration defaults
    default_warmup_days: int = 365
    default_max_evals: int = 50000
    max_concurrent_calibrations: int = 2
    
    # CORS
    cors_origins: list = ["http://localhost:3000", "http://localhost:5173"]
    
    # pyrrm path (for development, can mount pyrrm library)
    pyrrm_path: Optional[str] = None

    # Batch analysis: directory where batch result folders are placed on the server.
    # Each subfolder = one batch. Override via BATCH_RESULTS_DIR env var.
    batch_results_dir: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.uploads_dir, self.reports_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.get_batch_results_dir().mkdir(parents=True, exist_ok=True)

    def get_batch_results_dir(self) -> Path:
        """Absolute path to the directory where batch result folders are placed."""
        if self.batch_results_dir:
            return Path(self.batch_results_dir).resolve()
        # Default: pyrrm-gui/batch_results/ (sibling to backend/)
        return Path(__file__).resolve().parents[2] / "batch_results"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
