"""How sure the robot is of what it knows, and what it learned to do: the pure logic behind provenance, forgetting and know-how.

Memory types (see ontology.py): every fact about the world (semantic memory) records HOW it was learned, and its confidence fades
when it is not confirmed again (forgetting). Procedural memory records how the robot got from one place to the next (a Route: the
moves it made) and how its tricks went in each place. The graph code stores these; this module decides what they mean.

  seen      the camera detected it (the strongest evidence)
  told      a person said so ("the keys are in the drawer"): believed, but weaker until the camera confirms it
  inferred  the robot worked it out (a look-alike appeared elsewhere: maybe it moved): a hypothesis, the weakest
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

SOURCES = ("seen", "told", "inferred")
BASE_CONFIDENCE = {"seen": 1.0, "told": 0.6, "inferred": 0.4}
PHRASE = {"seen": "I saw it", "told": "someone told me", "inferred": "my guess"}


def _parse(ts) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def belief(source: str | None, last_confirmed, half_life_days: float, now: datetime | None = None,
           base: float | None = None) -> dict:
    """What a fact is worth now: its source, a confidence that halves every `half_life_days` without confirmation, and a word for
    how to say it. `base` overrides the source's starting confidence (e.g. the detector's own confidence for a sighting)."""
    src = source if source in SOURCES else "seen"  # facts written before provenance existed all came from the camera
    start = BASE_CONFIDENCE[src] if base is None else max(0.0, min(1.0, float(base)))
    t, now = _parse(last_confirmed), now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - t).total_seconds() / 86400.0) if t else 0.0
    conf = start * (0.5 ** (age_days / half_life_days)) if half_life_days > 0 else start
    if conf >= 0.7:
        say = "sure"
    elif conf >= 0.4:
        say = "probably"
    else:
        say = "not sure any more"
    return {"learned_from": src, "how": PHRASE[src], "confidence": round(conf, 2), "certainty": say,
            "days_since_confirmed": round(age_days, 1)}


def best_belief(evidence: list[tuple[str, object, float | None]], half_life_days: float, now: datetime | None = None) -> dict:
    """The strongest of several pieces of evidence for one fact, e.g. [("seen", last_seen, detector_conf), ("told", told_at, None)]:
    a sighting last week can be weaker than being told this morning. Evidence without a time is skipped."""
    best = None
    for src, ts, base in evidence:
        if _parse(ts) is None:
            continue
        b = belief(src, ts, half_life_days, now, base)
        if best is None or b["confidence"] > best["confidence"]:
            best = b
    return best or {"learned_from": None, "how": "no evidence", "confidence": 0.0, "certainty": "unknown", "days_since_confirmed": None}


# ---- procedural memory: the moves between two places
TURN = ("turn_left", "turn_right")


def motion_kind(x: float, yaw: float) -> str:
    """What a drive command does, as one word (walking wins over turning: a walk with a slight turn is still a walk)."""
    if x > 0:
        return "forward"
    if x < 0:
        return "back"
    if yaw > 0:
        return "turn_left"
    if yaw < 0:
        return "turn_right"
    return "idle"


def compress_journey(segments: list[dict], min_s: float = 0.25) -> list[dict]:
    """Drive segments [{kind, speed, seconds}] -> the steps of a route: idle time and blips dropped, repeats of the same move merged
    (holding a key sends a stream of commands; a short pause between two presses of W is still one walk)."""
    steps: list[dict] = []
    for s in segments:
        kind, secs = s.get("kind"), float(s.get("seconds") or 0.0)
        if kind not in ("forward", "back", *TURN) or secs < min_s:
            continue
        speed = int(s.get("speed") or 2)
        if steps and steps[-1]["kind"] == kind and steps[-1]["speed"] == speed:
            steps[-1]["seconds"] = round(steps[-1]["seconds"] + secs, 1)
        else:
            steps.append({"kind": kind, "speed": speed, "seconds": round(secs, 1)})
    return steps


def route_seconds(steps: list[dict]) -> float:
    return round(sum(float(s.get("seconds") or 0.0) for s in steps), 1)


def reverse_steps(steps: list[dict]) -> list[dict]:
    """The way back: turn round, then the same moves in reverse order with left and right swapped (walking stays walking)."""
    swap = {"turn_left": "turn_right", "turn_right": "turn_left"}
    back = [{**s, "kind": swap.get(s["kind"], s["kind"])} for s in reversed(steps)]
    return [{"kind": "turn_left", "speed": 3, "seconds": 0.0, "degrees": 180}] + back


WORDS = {"forward": "walk forward", "back": "back up", "turn_left": "turn left", "turn_right": "turn right"}


def describe_steps(steps: list[dict]) -> str:
    """'walk forward ~3 s, turn right ~1 s, walk forward ~2 s' (seconds are rough: the speed level and the floor change them)."""
    if not steps:
        return "no moves recorded"
    return ", ".join("turn round" if s.get("degrees") == 180 else
                     f"{WORDS.get(s['kind'], s['kind'])} ~{max(1, math.floor(float(s['seconds']) + 0.5))} s" for s in steps)


# ---- procedural memory: how a trick goes in a place
def trick_advice(attempts: int, ok: int, failed: int, avoid_after: int) -> dict:
    """Should the robot try this trick here? `failed` counts tries that went wrong (it fell, or someone said so). It avoids the
    trick once it failed `avoid_after` times here AND more often than it worked: a floor that makes it slip is a fact about the room."""
    avoid = avoid_after > 0 and failed >= avoid_after and failed > ok
    if avoid:
        why = f"it went wrong {failed} of {attempts} times here"
    elif failed:
        why = f"it went wrong {failed} of {attempts} times here, so careful"
    elif attempts:
        why = f"it worked every time here ({attempts}x)"
    else:
        why = "never tried here"
    return {"attempts": attempts, "ok": ok, "failed": failed, "avoid": avoid, "why": why}
