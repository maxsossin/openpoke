"""Signal vs. noise scoring.

A newsletter fact is 'novel' when it adds information not already in the
knowledge graph AND is not just pile-on repetition of a topic that has been
saturated in recent newsletter coverage.

Algorithm:
    1. Semantic similarity against recent framing embeddings for the same topic's
       stories, using the story_framings ChromaDB collection. Two framings that
       express the same idea in different words score as near-duplicate even when
       difflib would treat them as novel.
    2. Lexical fallback via difflib.SequenceMatcher when ChromaDB is unavailable
       or the framing collection is empty.
    3. Repetition penalty: if the topic has been mentioned many times across
       sources in the last 7 days, reduce novelty even when the wording differs.

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
import threading
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

# Cosine similarity threshold above which two framings are treated as
# semantically equivalent (same idea, different words). Lower than the entity
# synonym threshold (0.88) because framing texts are longer and naturally vary.
_FRAMING_SEMANTIC_THRESHOLD = 0.92

_framing_chroma_collection = None
_framing_chroma_lock = threading.Lock()


def _get_framing_collection():
    """Get (or lazily create) the ChromaDB collection for framing embeddings.

    Uses double-checked locking, matching the pattern in store.py._get_kg_collection.
    Returns None when ChromaDB is unavailable; callers fall back to difflib.
    """
    global _framing_chroma_collection
    if _framing_chroma_collection is None:
        with _framing_chroma_lock:
            if _framing_chroma_collection is None:
                try:
                    from ..shared_chroma import get_chroma_client
                    client = get_chroma_client()
                    if client is not None:
                        _framing_chroma_collection = client.get_or_create_collection(
                            "story_framings",
                            metadata={"hnsw:space": "cosine"},
                        )
                except Exception as exc:
                    logger.debug(
                        "Story framing ChromaDB unavailable; semantic novelty disabled: %s",
                        exc,
                    )
    return _framing_chroma_collection


def index_framing_text(framing_id: int, framing_text: str, story_node_id: int) -> None:
    """Index a stored framing into ChromaDB for future semantic novelty queries.

    Called from insert_story_framing (store_ext.py) after the SQLite write
    succeeds. Best-effort: failures are silently swallowed because the framing
    is already persisted in SQLite.

    framing_id is used as the ChromaDB document ID; it is the same value
    returned by insert_story_framing and uniquely identifies each framing row.
    story_node_id is stored as metadata so _semantic_framing_score can filter
    comparisons to the same story, preventing cross-story false near-duplicates.
    """
    try:
        collection = _get_framing_collection()
        if collection is None:
            return
        collection.upsert(
            ids=[str(framing_id)],
            documents=[framing_text],
            metadatas=[{"story_node_id": str(story_node_id)}],
        )
    except Exception as exc:
        logger.debug("Framing ChromaDB indexing failed [id=%s]: %s", framing_id, exc)


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
    story_node_id: Optional[int] = None,
) -> float:
    """Novelty score for a topic using framing similarity and coverage frequency.

    When framing_text is provided:
      1. ChromaDB semantic comparison: query the story_framings collection for
         the nearest existing framing embedding within the same story (filtered
         by story_node_id). If cosine similarity ≥ _FRAMING_SEMANTIC_THRESHOLD
         (0.92), the framings are semantically equivalent and the score is
         overridden to 1.0 - similarity (near zero). This catches paraphrases
         that difflib would score as novel.
         When story_node_id is None (story not yet created), falls back to
         difflib immediately to avoid cross-story false near-duplicates.
      2. Lexical fallback (difflib): used when ChromaDB is unavailable, the
         framing collection is empty, or story_node_id is not provided.
      3. Saturation penalty: applied after the similarity step.

    Framing data for the lexical fallback is sourced from kg_story_framings via
    a SQLite JOIN — blocking local I/O only, no external calls.

    Fallback: when framing_text is absent (empty or None), frequency-only scoring
    is applied and the fallback is logged explicitly.
    """
    recent_count = nl_store.get_topic_recent_mention_count(topic_node_id, days=7)

    if framing_text and framing_text.strip():
        # Step 1: attempt semantic comparison via ChromaDB
        semantic_score = _semantic_framing_score(framing_text, story_node_id=story_node_id)
        if semantic_score is not None:
            # ChromaDB returned a valid similarity — apply saturation and return
            if recent_count > 0:
                saturation = min(recent_count / _SATURATION_COUNT, 1.0)
                semantic_score *= 1.0 - 0.75 * saturation
            return max(0.0, min(1.0, semantic_score))

        # Step 2: ChromaDB unavailable or empty — fall back to lexical comparison.
        # Compare against all recent framings and take the minimum novelty score
        # (the existing framing most similar to the new one). Saturation is applied
        # once after the minimum is found, not inside each comparison call.
        existing_framings: List[str] = nl_store.get_recent_framings_for_topic(
            topic_node_id
        )
        if existing_framings:
            novelty_scores = [
                compute_fact_novelty_score(
                    new_value=framing_text,
                    existing_value=ef,
                    recent_newsletter_mentions=0,
                )
                for ef in existing_framings
            ]
            base_novelty = min(novelty_scores)
            if recent_count > 0:
                saturation = min(recent_count / _SATURATION_COUNT, 1.0)
                base_novelty = max(0.0, base_novelty * (1.0 - 0.75 * saturation))
            return base_novelty
        return compute_fact_novelty_score(
            new_value=framing_text,
            existing_value=None,
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


def _semantic_framing_score(
    framing_text: str,
    story_node_id: Optional[int] = None,
) -> Optional[float]:
    """Query ChromaDB for the nearest framing embedding and return novelty score.

    Returns a novelty score in [0.0, 1.0] computed from cosine similarity, or
    None when ChromaDB is unavailable, the collection contains no framings,
    or story_node_id is None (which forces the difflib fallback in the caller
    to avoid comparing against framings from unrelated stories).

    When story_node_id is provided, the query is filtered to framings from the
    same story so cross-story false near-duplicates cannot suppress genuine
    novelty.

    Caller is responsible for applying the saturation penalty.
    """
    if story_node_id is None:
        return None
    try:
        collection = _get_framing_collection()
        if collection is None or collection.count() == 0:
            return None
        results = collection.query(
            query_texts=[framing_text],
            n_results=1,
            where={"story_node_id": str(story_node_id)},
            include=["distances"],
        )
        distances = (results.get("distances") or [[]])[0]
        if not distances:
            return None
        similarity = 1.0 - distances[0]
        if similarity >= _FRAMING_SEMANTIC_THRESHOLD:
            # Semantic near-duplicate: score proportional to distance from threshold
            score = 1.0 - similarity  # e.g. similarity=0.95 → score=0.05
            logger.debug(
                "Semantic framing near-duplicate detected",
                extra={
                    "similarity": round(similarity, 4),
                    "novelty_score_before_saturation": round(score, 4),
                },
            )
            return score
        # Semantically distinct — return full novelty (saturation applied by caller)
        return 1.0 - similarity * 0.5  # graded: higher similarity → some penalty
    except Exception as exc:
        logger.debug("Framing ChromaDB query failed, falling back to difflib: %s", exc)
        return None


__all__ = [
    "NOVELTY_THRESHOLD",
    "compute_fact_novelty_score",
    "compute_topic_novelty",
    "index_framing_text",
]
