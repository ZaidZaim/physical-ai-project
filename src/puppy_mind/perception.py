"""Perception: JPEG frame -> structured Scene (objects, attributes, spatial relations).

PuppyPi's Pi is too small for a good detector, so frames are interpreted off-board with an
OpenAI vision model using structured outputs. The model never sees memory, only the frame,
so perception stays independent from (and cannot be biased by) what the graph already holds.
"""
from __future__ import annotations

import base64
import io

from openai import OpenAI
from PIL import Image

from .config import Settings
from .models import ROOM_TYPES, Scene

PROMPT = """You are the perception module of a small quadruped robot with a single low camera.
Describe ONLY what is clearly visible in the frame as structured data.

Rules:
- One entry per physically distinct object. Two chairs are two entries.
- label: singular, lowercase noun. Prefer these labels when they fit: {vocab}.
  {vocab_rule}
- color: dominant colour or "unknown". description: <=12 words that would help recognise THIS
  object again (markings, shape, size), or "".
- zone: left / center / right third of the frame.
- confidence: your honest 0-1 certainty the object is really present. Do not pad with guesses.
- relations: only spatial relations you can actually see, between objects in this list.
  ON (resting on top of), INSIDE, NEXT_TO (side by side, touching/adjacent), NEAR,
  IN_FRONT_OF, BEHIND. Subject/object are idx values. Prefer ON/INSIDE/NEXT_TO over NEAR.
- The camera has no depth sense: do not claim distances in metres.
- People: output a single entry with label "person". {people_rule}
- room_type: what kind of place this view shows, judged from fixtures and furniture (fridge, stove, sofa, bed, toilet, stairs...)
  not from the objects you list. Use one of: kitchen, living room, dining room, bedroom, bathroom, hallway, office, garage,
  outdoor, other, unknown. A view of only a wall or floor is "unknown" with low room_confidence.
- If the image is blurry, dark, or empty, return few/no objects and say so in the summary.
"""


def _prepare(jpeg: bytes, max_side: int = 1024) -> str:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Perception:
    def __init__(self, s: Settings):
        self.s = s
        self.client = OpenAI(api_key=s.openai_api_key)

    def describe(self, jpeg: bytes) -> Scene:
        s = self.s
        prompt = PROMPT.format(
            vocab=", ".join(s.vocab),
            vocab_rule=(
                "Use ONLY these labels; ignore everything else."
                if s.strict_vocab
                else "Other labels are allowed for clearly visible, relevant objects."
            ),
            people_rule=(
                "You may add a brief neutral description (clothing colour only)."
                if s.people_consent
                else "Do NOT describe the person: description must be '' and color 'unknown'. "
                "Never identify anyone."
            ),
        )
        resp = self.client.chat.completions.parse(
            model=s.vision_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this frame."},
                        {"type": "image_url", "image_url": {"url": _prepare(jpeg), "detail": "auto"}},
                    ],
                },
            ],
            response_format=Scene,
            temperature=0,
        )
        scene = resp.choices[0].message.parsed
        if scene is None:
            raise RuntimeError(f"vision model refused: {resp.choices[0].message.refusal}")
        return self._post(scene)

    def _post(self, scene: Scene) -> Scene:
        """Deterministic guardrails after the model: vocab filter, privacy, re-indexing."""
        s = self.s
        keep = [o for o in scene.objects if not s.strict_vocab or o.label.lower() in s.vocab]
        for o in keep:
            o.label = o.label.strip().lower()
            o.color = o.color.strip().lower() or "unknown"
            o.confidence = max(0.0, min(1.0, o.confidence))
            if o.label == "person" and not s.people_consent:
                o.description, o.color = "", "unknown"
        remap = {o.idx: i for i, o in enumerate(keep)}
        for o in keep:
            o.idx = remap[o.idx]
        rels = [
            r.model_copy(update={"subject": remap[r.subject], "object": remap[r.object]})
            for r in scene.relations
            if r.subject in remap and r.object in remap
        ]
        room = (scene.room_type or "unknown").strip().lower()
        room = room if room in ROOM_TYPES else "other"
        return Scene(summary=scene.summary, room_type=room, room_confidence=max(0.0, min(1.0, scene.room_confidence)),
                     objects=keep, relations=rels)
