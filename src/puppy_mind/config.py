from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# The project's .env must WIN over variables already in the shell/Windows environment: a stale NEO4J_URI from another
# project silently pointed this app at a different database. (Set env vars explicitly in .env to change behaviour.)
load_dotenv(override=True)


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
    nams_workspace_id: str = os.getenv("NAMS_WORKSPACE_ID", "")

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
    # a frame only counts as evidence of absence if it shares at least this fraction of the objects known (present) at the place
    min_overlap: float = float(os.getenv("MIN_OVERLAP", "0.5"))
    auto_place: bool = _bool("AUTO_PLACE", True)
    # horizontal field of view of the camera, used to turn 'left/centre/right third of the frame' into a bearing. A guess until measured.
    camera_fov_deg: float = float(os.getenv("CAMERA_FOV_DEG", "70"))
    place_confirm_frames: int = int(os.getenv("PLACE_CONFIRM_FRAMES", "2"))
    # agreeing frames a place never seen before needs. 2 (like a move): replaying the real tour, 3 lost a short bathroom visit
    # and 2 made no false room once the start-up rule, the same-place check and the moving/vague labels were in
    new_place_confirm_frames: int = int(os.getenv("NEW_PLACE_CONFIRM_FRAMES", "2"))
    # rooms are learned as neighbours only if the robot got from one to the other this quickly (else it may have passed through others)
    adjacency_max_gap_s: float = float(os.getenv("ADJACENCY_MAX_GAP_S", "60"))

    people_consent: bool = _bool("PEOPLE_CONSENT", False)
    store_images: bool = _bool("STORE_IMAGES", False)


def get_settings() -> Settings:
    return Settings()
