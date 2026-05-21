"""Cross-publication narrative threading.

Incoming newsletter content is stitched into an existing story node or spawns
a new one when two conditions are met:
    1. The newsletter provides a story_title (a focused, ongoing narrative).
    2. At least STORY_MIN_SOURCES distinct publications have covered the same
       primary topics in the last 14 days — preventing single-source stories
       from entering the graph.

Matching algorithm (no LLM):
    - Tokenise both story_title and existing story canonical_names.
    - Compute stop-word-filtered word overlap coefficient.
    - Threshold: ≥ STORY_MATCH_THRESHOLD → thread to existing story.
    - Below threshold and multi-source coverage exists → create new story.

Story nodes use node_type='story'. Framings are stored in kg_story_framings
(a separate table — never in kg_edges, which can only hold one active edge
per from/to/type pair). Contradictory framings are preserved as distinct rows.

All writes to kg_nodes and kg_node_facts go through the existing
KnowledgeGraphStore API to honour the constraint that no feature writes
directly to the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..store import KnowledgeGraphStore, _resolve_node_name
from .store_ext import NewsletterGraphStore
from .extractor import NewsletterIntelligence
from ....logging_config import logger
from ...gmail.processing import ProcessedEmail

# Minimum distinct publications covering a shared topic before a story node
# is created. Entity synonym resolution (Fix 1) consolidates fragmented topic
# nodes, making the multi-source gate reliably reachable.
STORY_MIN_SOURCES = 2

# Word overlap coefficient threshold for matching a title to an existing story.
# Restored from the compensating value of 0.60; entity resolution reduces false
# misses that previously required the looser threshold.
STORY_MATCH_THRESHOLD = 0.75

_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "at", "to", "and", "or", "for", "on",
    "with", "by", "is", "are", "was", "were", "its", "it", "as", "has",
    "have", "from", "this", "that", "over", "under",
})


@dataclass
class StoryThreadResult:
    story_node_id: Optional[int]
    story_name: str
    was_created: bool
    framing_id: Optional[int]


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        w for w in text.lower().split()
        if w not in _STOP_WORDS and len(w) > 2
    )


def _word_overlap(a: str, b: str) -> float:
    """Overlap coefficient: intersection / min(|A|, |B|).

    Returns 0.0 when either title has fewer than 2 content words — a single
    shared word (e.g. "policy") would otherwise produce a spurious 1.0 score.
    Titles with only one content word fall through to the ChromaDB semantic
    fallback in _find_matching_story.
    """
    wa, wb = _content_words(a), _content_words(b)
    if len(wa) < 2 or len(wb) < 2:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _find_matching_story(
    story_title: str,
    kg_store: KnowledgeGraphStore,
) -> Optional[Tuple[int, str, float]]:
    """Search existing story nodes for one whose title overlaps story_title.

    Returns (node_id, canonical_name, overlap_score) or None.
    Uses the first content word for the initial substring search to keep
    the candidate set small, then scores all candidates.
    """
    words = story_title.split()
    # Use the longest word (likely most distinctive) for the initial lookup
    seed = max((w for w in words if w.lower() not in _STOP_WORDS),
               key=len, default=words[0] if words else story_title)

    candidates = kg_store.search_nodes(seed, limit=15)

    best: Optional[Tuple[int, str, float]] = None
    for c in candidates:
        # search_nodes returns all node_types; filter to stories
        node = kg_store.query_node_by_name(c["canonical_name"], node_type="story")
        if node is None:
            continue
        overlap = _word_overlap(story_title, c["canonical_name"])
        if overlap >= STORY_MATCH_THRESHOLD:
            if best is None or overlap > best[2]:
                best = (c["id"], c["canonical_name"], overlap)

    if best is not None:
        return best

    # Semantic fallback: word overlap failed — use ChromaDB embedding similarity.
    # Uses a lower threshold (0.80) than entity resolution (0.88) because story
    # titles legitimately vary more than entity names (e.g. "AI Governance" vs
    # "Artificial Intelligence Policy"). _resolve_node_name returns the existing
    # node's canonical_name when similarity >= threshold, else the input unchanged.
    resolved = _resolve_node_name("story", story_title, threshold=0.80)
    if resolved != story_title:
        node = kg_store.query_node_by_name(resolved, node_type="story")
        if node is not None:
            return (node["id"], node["canonical_name"], 0.80)

    return None


async def thread_story(
    *,
    email: ProcessedEmail,
    intelligence: NewsletterIntelligence,
    source_node_id: int,
    topic_node_ids: List[int],
    topic_novelty_scores: Dict[int, float],
    kg_store: KnowledgeGraphStore,
    nl_store: NewsletterGraphStore,
    extracted_at: str,
) -> StoryThreadResult:
    """Attempt to thread a newsletter into an existing story, or create a new one.

    Returns a StoryThreadResult. story_node_id is None when the story cannot
    yet be created (insufficient source coverage) and no match was found.

    Framing is always stored for matched stories, regardless of when the
    story node was created. The framing timeline is the primary signal —
    it is never deduplicated or collapsed.
    """
    if not intelligence.has_story or not intelligence.has_framing:
        if intelligence.has_story and not intelligence.has_framing:
            logger.warning(
                "Story title present but no framing; LLM response malformed",
                extra={"email_id": email.id, "story_title": intelligence.story_title},
            )
        return StoryThreadResult(
            story_node_id=None, story_name="", was_created=False, framing_id=None,
        )

    story_title = intelligence.story_title
    source_ts = email.timestamp.isoformat(timespec="seconds")
    story_novelty = min(topic_novelty_scores.values()) if topic_novelty_scores else 1.0

    # --- Step 1: Try to match an existing story ---
    match = _find_matching_story(story_title, kg_store)

    if match:
        story_node_id, story_name, overlap = match
        logger.debug(
            "Newsletter threaded to existing story",
            extra={
                "story": story_name,
                "overlap": round(overlap, 2),
                "email_id": email.id,
            },
        )
        was_created = False

    else:
        # --- Step 2: Guard — need STORY_MIN_SOURCES before creating ---
        max_sources = 0
        for topic_id in topic_node_ids:
            stats = nl_store.get_topic_window_stats(
                topic_id, recent_days=14, prior_days=0,
            )
            max_sources = max(max_sources, stats["recent_sources"])

        if max_sources < STORY_MIN_SOURCES:
            logger.info(
                "Story creation deferred: insufficient source coverage",
                extra={
                    "story_title": story_title,
                    "sources_so_far": max_sources,
                    "required": STORY_MIN_SOURCES,
                },
            )
            nl_store.store_pending_framing(
                story_title=story_title,
                source_node_id=source_node_id,
                framing_text=intelligence.framing_text,
                sentiment=intelligence.sentiment,
                source_email_id=email.id,
                source_email_ts=source_ts,
                key_claim=intelligence.key_claim,
            )
            return StoryThreadResult(
                story_node_id=None,
                story_name=story_title,
                was_created=False,
                framing_id=None,
            )

        # --- Step 3: Create story node ---
        story_node_id = kg_store.get_or_create_node("story", story_title)
        story_name = story_title
        was_created = True

        kg_store.upsert_fact(
            node_id=story_node_id,
            fact_key="first_seen_at",
            fact_value=source_ts,
            confidence=1.0,
            source_email_id=email.id,
            source_email_timestamp=source_ts,
            extracted_at=extracted_at,
            novelty_score=story_novelty,
        )
        kg_store.upsert_fact(
            node_id=story_node_id,
            fact_key="status",
            fact_value="developing",
            confidence=0.9,
            source_email_id=email.id,
            source_email_timestamp=source_ts,
            extracted_at=extracted_at,
            novelty_score=story_novelty,
        )

        # Link story → each novel topic
        for topic_id in topic_node_ids:
            kg_store.upsert_edge(
                from_node_id=story_node_id,
                to_node_id=topic_id,
                edge_type="involves",
                confidence=0.9,
                source_email_id=email.id,
                source_email_timestamp=source_ts,
                extracted_at=extracted_at,
                novelty_score=topic_novelty_scores.get(topic_id, story_novelty),
            )

        logger.info(
            "Story node created",
            extra={
                "story": story_name,
                "email_id": email.id,
                "source_coverage": max_sources,
            },
        )

        # Flush framings held while source coverage was below threshold.
        # Ordered by created_at ASC so the timeline is chronologically correct.
        # novelty_score defaults to 1.0: original scores are not persisted in
        # kg_pending_framings, and these sources were genuinely early/novel.
        for pf in nl_store.pop_pending_framings(story_title):
            nl_store.insert_story_framing(
                story_node_id=story_node_id,
                source_node_id=pf["source_node_id"],
                framing_text=pf["framing_text"],
                sentiment=pf["sentiment"],
                source_email_id=pf["source_email_id"],
                source_email_ts=pf["source_email_ts"],
                key_claim=pf["key_claim"],
            )
            kg_store.upsert_edge(
                from_node_id=pf["source_node_id"],
                to_node_id=story_node_id,
                edge_type="covers",
                confidence=0.95,
                source_email_id=pf["source_email_id"],
                source_email_timestamp=pf["source_email_ts"],
                extracted_at=extracted_at,
            )

    # --- Step 4: Record this source's framing (always) ---
    framing_id = nl_store.insert_story_framing(
        story_node_id=story_node_id,
        source_node_id=source_node_id,
        framing_text=intelligence.framing_text,
        sentiment=intelligence.sentiment,
        source_email_id=email.id,
        source_email_ts=source_ts,
        key_claim=intelligence.key_claim,
    )

    # Mark source as covering this story (one active edge per source per story)
    kg_store.upsert_edge(
        from_node_id=source_node_id,
        to_node_id=story_node_id,
        edge_type="covers",
        confidence=0.95,
        source_email_id=email.id,
        source_email_timestamp=source_ts,
        extracted_at=extracted_at,
        novelty_score=story_novelty,
    )

    return StoryThreadResult(
        story_node_id=story_node_id,
        story_name=story_name,
        was_created=was_created,
        framing_id=framing_id,
    )


__all__ = ["StoryThreadResult", "STORY_MIN_SOURCES", "thread_story"]
