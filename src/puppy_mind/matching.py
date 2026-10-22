"""Entity identity: decide whether a detection is a known Object or a new one.

Pure functions (no I/O) so the riskiest logic in the project is unit-testable.
A wrong match writes a misleading memory, so we are conservative: an unmatched
detection becomes a new Object, and duplicates can be merged later.
"""
from __future__ import annotations

from dataclasses import dataclass

LABEL_BASE = 0.40
COLOR_MATCH = 0.25
COLOR_CLASH = -0.25
ZONE_MATCH = 0.10
ANCHOR_MATCH = 0.25
MATCH_THRESHOLD = 0.35


@dataclass
class Known:
    id: str
    label: str
    color: str
    zone: str
    anchors: frozenset[tuple[str, str]]  # {(predicate, other_label)} currently active


@dataclass
class Seen:
    idx: int
    label: str
    color: str
    zone: str
    confidence: float
    anchors: frozenset[tuple[str, str]]


# Colour words a vision model uses interchangeably for one thing (a 'white' cabinet in one frame is 'beige' or 'brown' in the next).
FAMILIES = [{"white", "cream", "ivory", "beige", "off-white"}, {"brown", "tan", "beige", "wood", "wooden"},
            {"black", "dark", "charcoal", "grey", "gray", "dark gray", "dark grey"}, {"transparent", "clear", "glass"}]
COLOR_FAMILY = 0.15


def _same_family(a: str, b: str) -> bool:
    return any(a in f and b in f for f in FAMILIES)


def _known(c: str) -> bool:
    return c not in ("", "unknown", "none")


def score(k: Known, s: Seen) -> float:
    if k.label != s.label:
        return -1.0
    sc = LABEL_BASE
    if _known(k.color) and _known(s.color):
        sc += COLOR_MATCH if k.color == s.color else (COLOR_FAMILY if _same_family(k.color, s.color) else COLOR_CLASH)
    if k.zone == s.zone and s.zone != "unknown":
        sc += ZONE_MATCH
    if k.anchors & s.anchors:
        sc += ANCHOR_MATCH
    return sc


def assign(knowns: list[Known], seens: list[Seen]) -> dict[int, str | None]:
    """Greedy best-first one-to-one assignment. Returns {detection idx: object id or None}."""
    pairs = sorted(
        ((score(k, s), k.id, s.idx) for k in knowns for s in seens),
        key=lambda t: t[0],
        reverse=True,
    )
    out: dict[int, str | None] = {s.idx: None for s in seens}
    used: set[str] = set()
    for sc, kid, sidx in pairs:
        if sc < MATCH_THRESHOLD:
            break
        if kid in used or out[sidx] is not None:
            continue
        out[sidx] = kid
        used.add(kid)
    return out
