from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_user: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

    nams_api_key: str = os.getenv("NAMS_API_KEY", "")
    nams_base_url: str = os.getenv("NAMS_BASE_URL", "https://memory.neo4jlabs.com")

    robot_name: str = os.getenv("ROBOT_NAME", "PuppyPi")
    default_location: str = os.getenv("DEFAULT_LOCATION", "Demo Area")
    camera_source: str = os.getenv("CAMERA_SOURCE", "")
    watch_interval_s: float = float(os.getenv("WATCH_INTERVAL_S", "5"))
    brain_token: str = os.getenv("BRAIN_TOKEN", "")

    vocab: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            v.strip().lower()
            for v in os.getenv(
                "OBJECT_VOCAB",
                "laptop,bottle,cup,chair,table,backpack,phone,book,monitor,keyboard,plant,box",
            ).split(",")
            if v.strip()
        )
    )
    strict_vocab: bool = _bool("STRICT_VOCAB", False)
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.5"))
    miss_threshold: int = int(os.getenv("MISS_THRESHOLD", "3"))

    people_consent: bool = _bool("PEOPLE_CONSENT", False)
    store_images: bool = _bool("STORE_IMAGES", False)


def get_settings() -> Settings:
    return Settings()
