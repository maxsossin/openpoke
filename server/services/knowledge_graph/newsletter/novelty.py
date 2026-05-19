"""Signal vs. noise scoring.

A newsletter fact is 'novel' when it adds information not already in the
knowledge graph AND is not just pile-on repetition of a topic that has been
saturated in recent newsletter coverage.

Algorithm (no external dependencies — uses stdlib difflib):
    1. Lexical similarity against the existing fact value for the same node+key.
       Near-identical text → near-zero novelty score.
    2. Repetition penalty: if the topic has been mentioned many times across
       sources in the last 7 days, reduce novelty even when the wording differs.
       Different sources often paraphrase the same underlying fact.

Thresholds are calibrated conservatively. The user's goal is less information
that is more valuable, so borderline facts are treated as low-novelty.

Scoring scale:
    0.0  — pure repetition: identical text or completely saturated topic
    0.30 — NOVELTY_THRESHOLD: facts below this are excluded from momentum
           tracking and story creation
    1.0  — genuinely novel: new information not in the graph, low recent coverage
"""

from __future__ import annotations

import difflib
from typing import List, Optional

from .store_ext import NewsletterGraphStore
from ....logging_config import logger

# Facts below this score are treated as noise for downstream processing.
# Momentum and story threading only receive facts above this threshold.
NOVELTY_THRESHOLD = 0.30

# Recent mention count at or above this level saturates novelty.
# 8 novel mentions in 7 days (≈ one per day) means the topic is well-covered.
_SATURATION_COUNT = 8

# Lexical similarity thresholds
_SIMILARITY_NOISE = 0.85   # at or above → nearly identical, novelty ≈ 0
_SIMILARITY_PARTIAL = 0.50  # above this → partial overlap, graded penalty


def compute_fact_novelty_score(
    new_value: str,
    existing_value: Optional[str],
    recent_newsletter_mentions: int = 0,
) -> float:
    """Score novelty of a single fact value in [0.0, 1.0].

    new_value:                  the fact_value just extracted
    existing_value:             current active fact_value for (node, key), or None
    recent_newsletter_mentions: total novel-weight mentions of this topic in last 7 days
    """
    score = 1.0

    # Step 1: lexical comparison against existing graph value
    if existing_value is not None:
        ratio = difflib.SequenceMatcher(
            None,
            new_value.lower().strip(),
            existing_value.lower().strip(),
        ).ratio()

        if ratio >= _SIMILARITY_NOISE:
            # Text is essentially identical — pure repetition
            score = 1.0 - ratio
        elif ratio >= _SIMILARITY_PARTIAL:
            # Significant overlap but not identical — graded penalty
            score = 1.0 - ratio * 0.75

    # Step 2: recent saturation penalty.
    # Even if the wording is new, a saturated topic adds minimal value.
    if recent_newsletter_mentions > 0:
        saturation = min(recent_newsletter_mentions / _SATURATION_COUNT, 1.0)
        # Up to 75% reduction at full saturation, ensuring a fully-saturated
        # topic (saturation=1.0) falls below NOVELTY_THRESHOLD (0.30).
        score *= 1.0 - 0.75 * saturation

    return max(0.0, min(1.0, score))


def compute_topic_novelty(
    topic_node_id: int,
    nl_store: NewsletterGraphStore,
    framing_text: Optional[str] = None,
) -> float:
    """Novelty score for a topic using framing similarity and coverage frequency.

    When framing_text is provided, the score compares the current newsletter's
    framing against the most recent framing for this topic's associated stories
    via compute_fact_novelty_score() (difflib lexical comparison). This ensures
    two newsletters covering the same topic from different angles are not treated
    as identical repetition.

    Framing data is sourced from kg_story_framings via a SQLite JOIN —
    blocking local I/O only, no external calls.

    Fallback: when framing_text is absent (empty or None), frequency-only scoring
    is applied and the fallback is logged explicitly.
    """
    recent_count = nl_store.get_topic_recent_mention_count(topic_node_id, days=7)

    if framing_text and framing_text.strip():
        existing_framings: List[str] = nl_store.get_recent_framings_for_topic(
            topic_node_id
        )
        existing_value = existing_framings[0] if existing_framings else None
        return compute_fact_novelty_score(
            new_value=framing_text,
            existing_value=existing_value,
            recent_newsletter_mentions=recent_count,
        )

    # Fallback: no framing text available — frequency-only scoring
    logger.debug(
        "Topic novelty: framing text unavailable, using frequency-only scoring",
        extra={"topic_node_id": topic_node_id},
    )
    return compute_fact_novelty_score(
        new_value="",
        existing_value=None,
        recent_newsletter_mentions=recent_count,
    )


__all__ = [
    "NOVELTY_THRESHOLD",
    "compute_fact_novelty_score",
    "compute_topic_novelty",
]
