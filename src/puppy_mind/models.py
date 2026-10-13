"""Structured observation schema: what perception returns and memory ingests.

All fields are required (no defaults) so the model works as an OpenAI
structured-output schema; use "" / "unknown" for unknowns.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Predicate = Literal["ON", "INSIDE", "NEXT_TO", "NEAR", "IN_FRONT_OF", "BEHIND"]
# ON/INSIDE are containment-style: an object has at most one active at a time.
EXCLUSIVE_PREDICATES = ("ON", "INSIDE")
PREDICATES = ("ON", "INSIDE", "NEXT_TO", "NEAR", "IN_FRONT_OF", "BEHIND")
Zone = Literal["left", "center", "right", "unknown"]


class Detection(BaseModel):
    idx: int = Field(description="Index of this object within this frame, starting at 0")
    label: str = Field(description="Singular lowercase noun, e.g. 'laptop', 'chair'")
    color: str = Field(description="Dominant colour, lowercase, or 'unknown'")
    description: str = Field(description="Short distinguishing description (<=12 words), or ''")
    zone: Zone = Field(description="Horizontal position in the frame")
    confidence: float = Field(description="0.0-1.0 certainty that this object is really there")


class Relation(BaseModel):
    subject: int = Field(description="idx of the subject object")
    predicate: Predicate
    object: int = Field(description="idx of the object the subject relates to")
    confidence: float


class Scene(BaseModel):
    summary: str = Field(description="One or two sentence description of the scene")
    objects: list[Detection]
    relations: list[Relation]
