"""Spatial memory, step one: "where am I?"

The robot has no map and no reliable odometry, so it recognises PLACES the way people do from a glance: what kind of room this looks
like, and which objects it already knows to live there. Pure logic (no I/O) so it can be tested exhaustively; brain.py wires it to memory.

  PlaceResolver  scores a view against every known place -> stay / switch / new place
  PlaceTracker   adds hysteresis so a single odd frame (a wall, a blur, someone walking by) cannot flip the robot's idea of where it is

Scoring, per candidate place:
  overlap  = share of the view that this place already explains, weighted by how DISTINCTIVE each object is
             (a fridge says "kitchen" far more than a chair does; weights come from how many places contain the label)
  room     = bonus if the view's room type matches the place's, penalty if both are known and differ
Colour is a soft cue: an exact colour match counts fully, a colour mismatch counts a little (lighting changes, view angle),
an unknown colour counts partially.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

UNINFORMATIVE_ROOMS = {"unknown", "other", ""}
# Objects found in almost every room carry little evidence about WHICH room this is.
COMMON_PRIOR = {"chair": 0.5, "table": 0.6, "door": 0.5, "box": 0.5, "bag": 0.6, "shoe": 0.6, "cable": 0.4, "person": 0.1, "cat": 0.3, "dog": 0.3}


@dataclass
class PlaceInfo:
    name: str
    room_type: str = "unknown"
    provisional: bool = False
    visits: int = 0
    objects: list[dict] = field(default_factory=list)  # {"label","color","seen","status"}


@dataclass
class Decision:
    place: str | None          # chosen existing place; None means "a new place"
    is_new: bool
    stay: bool
    confidence: float
    reason: str
    scores: dict[str, float]
    suggested_name: str | None = None


@dataclass
class TrackResult:
    place: str | None          # place to file this frame under (None only before any place exists and not yet created)
    changed: bool              # the robot moved to a different place (confirmed)
    is_new: bool               # ...and that place did not exist
    uncertain: bool            # a transition is suspected but not confirmed: do not count misses on this frame
    confidence: float
    reason: str
    suggested_name: str | None = None


def _color_match(a: str, b: str) -> float:
    a, b = (a or "unknown").lower(), (b or "unknown").lower()
    if a == b and a != "unknown":
        return 1.0
    if "unknown" in (a, b):
        return 0.7
    return 0.35  # same kind of object, different colour: weak but not zero (light, angle)


class PlaceResolver:
    def __init__(self, stay_threshold: float = 0.25, switch_threshold: float = 0.45, switch_margin: float = 0.2,
                 min_objects_for_new: int = 2, room_bonus: float = 0.15, room_penalty: float = 0.8):
        self.stay_threshold, self.switch_threshold, self.switch_margin = stay_threshold, switch_threshold, switch_margin
        self.min_objects_for_new = min_objects_for_new
        self.room_bonus, self.room_penalty = room_bonus, room_penalty

    # ---- scoring
    @staticmethod
    def _idf(places: list[PlaceInfo]) -> dict[str, float]:
        n = max(1, len(places))
        df: dict[str, int] = {}
        for p in places:
            for lab in {o["label"].lower() for o in p.objects}:
                df[lab] = df.get(lab, 0) + 1
        return {lab: math.log(1 + n / (1 + c)) + 0.5 for lab, c in df.items()}

    def _weight(self, label: str, idf: dict[str, float]) -> float:
        return idf.get(label, math.log(1 + 1) + 0.5) * COMMON_PRIOR.get(label, 1.0)

    def score(self, view: list[dict], room: str, room_conf: float, place: PlaceInfo, idf: dict[str, float]) -> float:
        known: dict[str, list[tuple[str, float]]] = {}
        for o in place.objects:
            w = 0.5 if o.get("status") == "possibly_gone" else 1.0
            known.setdefault(o["label"].lower(), []).append(((o.get("color") or "unknown").lower(), w))
        total = explained = 0.0
        for v in view:
            lab = v["label"].lower()
            wt = self._weight(lab, idf)
            total += wt
            cands = known.get(lab)
            if cands:
                explained += wt * max(_color_match(v.get("color", "unknown"), c) * w for c, w in cands)
        overlap = explained / total if total else 0.0
        s = overlap
        if room not in UNINFORMATIVE_ROOMS and place.room_type not in UNINFORMATIVE_ROOMS:
            if room == place.room_type:
                s += self.room_bonus * room_conf
            else:  # a kitchen view cannot be the living room, however many chairs both have: strong veto, scaled by confidence
                s *= 1.0 - self.room_penalty * room_conf
        return max(0.0, min(1.0, s))

    # ---- decision
    def decide(self, view: list[dict], room: str, room_conf: float, places: list[PlaceInfo], current: str | None) -> Decision:
        room = (room or "unknown").lower()
        informative = bool(view) or (room not in UNINFORMATIVE_ROOMS and room_conf >= 0.5)
        idf = self._idf(places)
        scores = {p.name: round(self.score(view, room, room_conf, p, idf), 3) for p in places}
        if not informative:
            # A blank wall / blur: say nothing about where we are.
            return Decision(current, False, True, scores.get(current or "", 0.0), "view has no usable evidence: staying put", scores)
        if current and current in scores:
            cur = scores[current]
            others = {k: v for k, v in scores.items() if k != current}
            best, best_s = (max(others.items(), key=lambda kv: kv[1]) if others else (None, 0.0))
            if best and best_s >= self.switch_threshold and best_s >= cur + self.switch_margin:
                return Decision(best, False, False, best_s, f"matches '{best}' ({best_s:.2f}) much better than '{current}' ({cur:.2f})", scores)
            if cur >= self.stay_threshold or (best_s < self.switch_threshold and len(view) < self.min_objects_for_new):
                return Decision(current, False, True, cur, f"still looks like '{current}' ({cur:.2f})", scores)
            cur_place = next((p for p in places if p.name == current), None)
            if (cur_place and room not in UNINFORMATIVE_ROOMS and room == cur_place.room_type and room_conf >= 0.5
                    and best_s < self.switch_threshold):
                # Same KIND of room and nothing else fits: this is another wall of the same room, not a second kitchen.
                return Decision(current, False, True, max(cur, 0.3), f"another view of '{current}' (same kind of room, nothing else matches)", scores)
        elif places:
            best, best_s = max(scores.items(), key=lambda kv: kv[1])
            if best_s >= self.switch_threshold:
                return Decision(best, False, False, best_s, f"recognised as '{best}' ({best_s:.2f})", scores)
        best_s = max(scores.values(), default=0.0)
        if len(view) >= self.min_objects_for_new or room not in UNINFORMATIVE_ROOMS:
            return Decision(None, True, False, 1.0 - best_s, f"matches no known place (best {best_s:.2f}): a new place", scores,
                            self.suggest_name(room, room_conf, places))
        return Decision(current, False, True, best_s, "too little evidence to leave the current place", scores)

    @staticmethod
    def suggest_name(room: str, room_conf: float, places: list[PlaceInfo]) -> str:
        taken = {p.name.lower() for p in places}
        base = room.title() if room not in UNINFORMATIVE_ROOMS and room_conf >= 0.5 else "Unnamed area"
        if base.lower() not in taken and base != "Unnamed area":
            return base
        n = 1 if base == "Unnamed area" else 2
        while f"{base} {n}".lower() in taken:
            n += 1
        return f"{base} {n}"


class PlaceTracker:
    """Hysteresis around PlaceResolver: a change (switch or new place) needs `confirm_frames` consecutive agreeing frames."""

    def __init__(self, resolver: PlaceResolver | None = None, confirm_frames: int = 2):
        self.resolver = resolver or PlaceResolver()
        self.confirm = max(1, confirm_frames)
        self.current: str | None = None
        self._pending: tuple | None = None
        self._count = 0

    def set_current(self, name: str | None) -> None:
        self.current, self._pending, self._count = name, None, 0

    def update(self, view: list[dict], room: str, room_conf: float, places: list[PlaceInfo]) -> TrackResult:
        d = self.resolver.decide(view, room, room_conf, places, self.current)
        if d.stay:
            self._pending, self._count = None, 0
            return TrackResult(self.current, False, False, False, d.confidence, d.reason)
        key = ("new",) if d.is_new else ("switch", d.place)
        self._count = self._count + 1 if key == self._pending else 1
        self._pending = key
        if self.current is None or self._count >= self.confirm:
            self._pending, self._count = None, 0
            if d.is_new:
                return TrackResult(None, True, True, False, d.confidence, d.reason, d.suggested_name)
            self.current = d.place
            return TrackResult(d.place, True, False, False, d.confidence, d.reason)
        return TrackResult(self.current, False, False, True, d.confidence,
                           f"{d.reason}; waiting for confirmation ({self._count}/{self.confirm})")

    def confirm_new(self, name: str) -> None:
        """Brain created the place the last result asked for."""
        self.current, self._pending, self._count = name, None, 0
