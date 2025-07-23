import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Configuration dataclass for the photo organizer application."""

    model_name: str = "gemma3:4b"
    photos_directory: str = "photos"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "gemma3:4b"),
            photos_directory=os.getenv("PHOTOS_DIRECTORY", "photos"),
        )


config = AppConfig.from_env()
