from __future__ import annotations
from typing import List
from pydantic import BaseModel
import re


class Fact(BaseModel):
    """Minimal unit extracted from text — a medical finding or statement."""
    text: str
    kind: str  # e.g. symptom, medication, plan, other


class Extracted(BaseModel):
    """Container for extracted facts and raw text."""
    text: str
    facts: List[Fact]


def extract(text: str) -> Extracted:
    """
    Simplified rule-based extractor that identifies key clinical entities:
    symptoms, medications, plans, etc.
    Replace this with spaCy / MedSpaCy / scispaCy for production use.
    """
    if not text:
        return Extracted(text="", facts=[])

    facts: List[Fact] = []

    # --- symptom detection ---
    for match in re.findall(r"\b(pain|fever|cough|fatigue|weakness|headache)\b", text, flags=re.I):
        facts.append(Fact(text=match.lower(), kind="symptom"))

    # --- medication detection ---
    for match in re.findall(r"\b(ibuprofen|acetaminophen|metformin|insulin|amoxicillin)\b", text, flags=re.I):
        facts.append(Fact(text=match.lower(), kind="med"))

    # --- plan / order detection ---
    for match in re.findall(r"\b(follow up|refer|prescribe|lab test|imaging|x-ray|therapy)\b", text, flags=re.I):
        facts.append(Fact(text=match.lower(), kind="order"))

    # fallback: if no patterns match, treat as a general statement
    if not facts:
        facts.append(Fact(text=text[:100], kind="other"))

    return Extracted(text=text, facts=facts)
