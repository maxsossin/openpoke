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
from typing import Optional

from .store_ext import NewsletterGraphStore

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
) -> float:
    """Novelty score for a topic node based solely on recent coverage frequency.

    Used when no existing fact value is available to compare against
    (e.g. the topic node is new, or only the topic's presence matters).
    """
    recent_count = nl_store.get_topic_recent_mention_count(topic_node_id, days=7)
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
