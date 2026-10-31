"""Speech: audio in -> text (OpenAI transcription), text -> audio out (OpenAI TTS).

The robot's WonderEcho Pro is just a USB microphone and speaker as far as this code is concerned: robot/puppy_voice.py records and plays,
the laptop does the understanding and the talking (same split as the camera: the Pi does no inference).
"""
from __future__ import annotations

import re

from .config import Settings

# Words a speech model tends to mangle; passed as a prompt so "Cypher" is not heard as "cipher".
VOCAB_HINT = "Neo4j, Cypher, GraphRAG, Aura, AuraDB, GenAI, LLM, knowledge graph, vector index, embeddings, Graph Data Science, MCP"


# What Whisper-style models "hear" in silence or hum: video outros and subtitle credits from their training data. A visitor at a robot
# booth never says these, so a transcript made of them means the microphone gave us nothing.
_SILENCE = re.compile(
    r"thanks? (you )?(so much )?for watching|subscribe|click the bell|see you (in the )?next (video|time)|"
    r"subtitles? (by|from)|amara\.org|transcribed by", re.I)


def heard_nothing(text: str) -> bool:
    """True when a transcript is empty, a lone filler word, or one of the phrases the model invents for silence."""
    t = re.sub(r"[^\w\s']", " ", (text or "").lower()).split()
    return not t or (len(t) == 1 and t[0] in ("you", "bye", "uh", "um", "hmm", "so", "okay")) or bool(_SILENCE.search(text or ""))


class Voice:
    def __init__(self, s: Settings, client=None):
        self.s = s
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=s.openai_api_key)
        self.client = client

    def transcribe(self, audio: bytes, filename: str = "speech.wav") -> str:
        r = self.client.audio.transcriptions.create(model=self.s.stt_model, file=(filename, audio), prompt=VOCAB_HINT,
                                                    language=self.s.voice_language or None)  # never guess: see Settings.voice_language
        return (getattr(r, "text", "") or "").strip()

    def speak(self, text: str) -> bytes:
        """WAV bytes, so the robot can play them with plain `aplay`."""
        r = self.client.audio.speech.create(model=self.s.tts_model, voice=self.s.tts_voice, input=text[:1500], response_format="wav")
        return r.content if hasattr(r, "content") else r.read()
