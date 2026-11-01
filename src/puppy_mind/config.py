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
    # booth conversation: greet people the camera sees, talk, and (only with a spoken yes) remember them as leads
    people_greet: bool = _bool("PEOPLE_GREET", True)
    greet_cooldown_s: float = float(os.getenv("GREET_COOLDOWN_S", "60"))  # at most one greeting per this many seconds, whoever it is for
    max_greetings_per_hour: int = int(os.getenv("MAX_GREETINGS_PER_HOUR", "40"))
    person_regreet_s: float = float(os.getenv("PERSON_REGREET_S", "1800"))  # the same badge name is not greeted again within this
    group_min: int = int(os.getenv("GROUP_MIN", "3"))  # this many engaged people get one group hello instead of one each
    # booth mode: look slightly up at people and greet with a trick (Hiwonder action group; '' for none)
    booth_mode: bool = _bool("BOOTH_MODE", False)
    look_up_deg: float = float(os.getenv("LOOK_UP_DEG", "10"))
    greet_action: str = os.getenv("GREET_ACTION", "")  # no trick when greeting by default; tricks still play on request
    # the robot only starts a conversation when it hears this; follow-ups need no wake word for WAKE_WINDOW_S. Empty: no wake word.
    wake_word: str = os.getenv("WAKE_WORD", "Hey PuppyPi")
    wake_window_s: float = float(os.getenv("WAKE_WINDOW_S", "90"))
    # on "Hey PuppyPi": this trick first, then the robot looks up at the visitor by ENGAGED_LOOK_UP_DEG for the conversation, and back
    # to IDLE_LOOK_UP_DEG when it goes quiet. Hiwonder's pose node rejects more than 31 deg of pitch, so it is capped at 30.
    wake_action: str = os.getenv("WAKE_ACTION", "boxing")
    engaged_look_up_deg: float = float(os.getenv("ENGAGED_LOOK_UP_DEG", "25"))
    idle_look_up_deg: float = float(os.getenv("IDLE_LOOK_UP_DEG", "0"))
    gesture_cooldown_s: float = float(os.getenv("GESTURE_COOLDOWN_S", "60"))
    max_gestures_per_hour: int = int(os.getenv("MAX_GESTURES_PER_HOUR", "30"))
    encounter_idle_s: float = float(os.getenv("ENCOUNTER_IDLE_S", "300"))
    stt_model: str = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")  # 2-3x faster than whisper-1 in tests
    tts_model: str = os.getenv("OPENAI_TTS_MODEL", "tts-1")
    tts_voice: str = os.getenv("OPENAI_TTS_VOICE", "alloy")
    # the booth language: speech is transcribed as this language (ISO 639-1) and the Wizard always answers in it. Without it the speech
    # model guesses per recording, and a short noisy clip can come out as another language, which the Wizard then answers in.
    voice_language: str = os.getenv("VOICE_LANGUAGE", "en")
    # what is really on at the booth/event (demos, talks, times). Without it the Wizard says to ask the staff instead of inventing it.
    booth_info: str = os.getenv("BOOTH_INFO", "")
    event_name: str = os.getenv("EVENT_NAME", "the conference")
    store_images: bool = _bool("STORE_IMAGES", False)


def get_settings() -> Settings:
    return Settings()
