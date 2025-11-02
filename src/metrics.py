from __future__ import annotations
from typing import List
from pydantic import BaseModel
from rapidfuzz import fuzz


class Scores(BaseModel):
    """Quantitative results for one transcript ↔ note pair."""
    missing_ct: int
    halluc_ct: int
    conflict_ct: int
    quality_score: float


def _fuzzy_in(fact, pool, threshold=90):
    """Check if a fact is semantically similar to any fact in a pool."""
    for g in pool:
        if fact.kind == g.kind and fuzz.partial_ratio(fact.text, g.text) >= threshold:
            return True
    return False


def score(transcript, note) -> Scores:
    """
    Compare facts extracted from transcript (truth) vs note (candidate).
    Returns counts of missing, hallucinated, and conflicting items.
    """
    ref = transcript.facts
    gen = note.facts

    missing = [r for r in ref if not _fuzzy_in(r, gen)]
    halluc = [g for g in gen if not _fuzzy_in(g, ref)]

    # placeholder conflict logic (extend with UMLS negation detection)
    conflicts = []
    for g in gen:
        for r in ref:
            if g.kind == r.kind and g.text != r.text and fuzz.ratio(g.text, r.text) < 40:
                conflicts.append(g)
                break

    qscore = max(0.0, 1.0 - (len(missing) + len(halluc) + len(conflicts)) / max(1, len(ref) + len(gen)))

    return Scores(
        missing_ct=len(missing),
        halluc_ct=len(halluc),
        conflict_ct=len(conflicts),
        quality_score=qscore,
    )
