"""Newsletter intelligence orchestrator.

Runs all four features as a single unified pipeline per email:
    Phase 0: Newsletter classification (heuristic, no LLM)
    Phase 1: Ensure publication source node
    Phase 2: Combined LLM extraction (replaces standard extract_from_email)
    Phase 3: Signal/noise gate — topic novelty scoring
    Phase 4: Story threading (novel topics only, story_title required)
    Phase 5: Contrarian detection (story with sufficient framing history)
    Phase 6: Periodic momentum check (every _MOMENTUM_CHECK_INTERVAL calls)

Sequencing guarantee:
    Phases 3–5 only receive topics with novelty_score > NOVELTY_THRESHOLD.
    Low-novelty topics do not enter momentum counts, story nodes, or
    contrarian analysis. This is the single enforcement point for the
    constraint that pure repetition must not inflate any downstream signal.

Integration contract:
    process_email() returns a NewsletterProcessingResult whose
    extraction_result field (Optional[ExtractionResult]) must be passed to
    KnowledgeGraphWatcher._apply_extraction(). All KG node/fact/edge writes
    flow through the existing store API — this processor never writes directly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from ..store import KnowledgeGraphStore
from ..extractor import ExtractionResult
from .store_ext import NewsletterGraphStore
from .classifier import NewsletterClassification, classify_newsletter
from .extractor import NewsletterIntelligence, extract_newsletter_intelligence
from .novelty import NOVELTY_THRESHOLD, compute_topic_novelty
from .story_threader import StoryThreadResult, thread_story
from .momentum import check_and_surface_momentum
from .contrarian import detect_contrarian_position
from ....logging_config import logger
from ...gmail.processing import ProcessedEmail

# Momentum check runs once every N calls to process_email.
# At 60-second poll intervals with ≤10 emails per poll, this fires
# roughly every 30 minutes — frequent enough to catch acceleration,
# conservative enough to avoid alert fatigue.
_MOMENTUM_CHECK_INTERVAL = 30


@dataclass
class NewsletterProcessingResult:
    email_id: str
    is_newsletter: bool
    source_name: str
    novel_topic_count: int
    extraction_result: Optional[ExtractionResult] = None
    story_result: Optional[StoryThreadResult] = None
    contrarian_status: Optional[str] = None
    contrarian_evidence_count: int = 0


class NewsletterIntelligenceProcessor:
    """Unified pipeline: classification → extraction → novelty → story → contrarian.

    Called from KnowledgeGraphWatcher._extract_batch() for every email.
    Returns None for non-newsletters; returns a NewsletterProcessingResult
    whose extraction_result is fed into the existing _apply_extraction() path.
    """

    def __init__(
        self,
        kg_store: KnowledgeGraphStore,
        nl_store: NewsletterGraphStore,
    ) -> None:
        self._kg_store = kg_store
        self._nl_store = nl_store
        self._call_count = 0

    async def process_email(
        self,
        email: ProcessedEmail,
    ) -> Optional[NewsletterProcessingResult]:
        """Process a single email through the newsletter intelligence pipeline.

        Returns None for non-newsletters so the caller can fall back to the
        standard extraction path. Never raises — failures in individual phases
        are isolated so a contrarian detection crash does not abort story
        threading.
        """
        self._call_count += 1

        # Phase 0: classify (fast heuristic — no LLM)
        classification = classify_newsletter(email)
        self._nl_store.record_newsletter_meta(
            email_id=email.id,
            source_node_id=None,
            is_newsletter=classification.is_newsletter,
        )

        if not classification.is_newsletter:
            return None

        logger.debug(
            "Newsletter detected",
            extra={
                "email_id": email.id,
                "publication": classification.publication_name,
                "signals": classification.signals,
            },
        )

        # Phase 1: ensure source node
        source_node_id = self._ensure_source_node(classification)
        self._nl_store.increment_source_email_count(source_node_id)
        self._nl_store.update_newsletter_meta_source(email.id, source_node_id)

        # Phase 2: newsletter-specific LLM extraction
        intelligence = await extract_newsletter_intelligence(email)
        if intelligence is None:
            return NewsletterProcessingResult(
                email_id=email.id,
                is_newsletter=True,
                source_name=classification.publication_name,
                novel_topic_count=0,
                extraction_result=None,
            )

        extracted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Phase 3: novelty scoring — gate for all downstream phases
        novel_topic_ids, novel_count = self._score_and_record_topics(
            intelligence=intelligence,
            source_node_id=source_node_id,
        )

        # Phase 4: story threading (only if novel topics exist and story title present)
        story_result: Optional[StoryThreadResult] = None
        if novel_topic_ids and not intelligence.has_story:
            logger.info(
                "Story threading skipped: LLM returned no story_title",
                extra={"email_id": email.id, "novel_topics": novel_topic_ids},
            )
        if novel_topic_ids and intelligence.has_story:
            try:
                story_result = await thread_story(
                    email=email,
                    intelligence=intelligence,
                    source_node_id=source_node_id,
                    topic_node_ids=novel_topic_ids,
                    kg_store=self._kg_store,
                    nl_store=self._nl_store,
                    extracted_at=extracted_at,
                )
            except Exception as exc:
                logger.exception(
                    "Story threading failed [email=%s]: %s", email.id, exc,
                )

        # Phase 5: contrarian detection (story must exist and have framing history)
        contrarian_status: Optional[str] = None
        contrarian_evidence = 0
        if (
            story_result is not None
            and story_result.story_node_id is not None
            and intelligence.has_framing
        ):
            try:
                result = await detect_contrarian_position(
                    story_node_id=story_result.story_node_id,
                    source_node_id=source_node_id,
                    intelligence=intelligence,
                    email=email,
                    nl_store=self._nl_store,
                )
                if result is not None:
                    contrarian_status, contrarian_evidence = result
            except Exception as exc:
                logger.exception(
                    "Contrarian detection failed [email=%s]: %s", email.id, exc,
                )

        # Phase 6: periodic momentum check (non-blocking background task)
        if self._call_count % _MOMENTUM_CHECK_INTERVAL == 0:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._run_momentum_check(),
                    name="newsletter-momentum-check",
                )
            except RuntimeError:
                pass  # no running loop; skip silently

        return NewsletterProcessingResult(
            email_id=email.id,
            is_newsletter=True,
            source_name=classification.publication_name,
            novel_topic_count=novel_count,
            extraction_result=intelligence.to_extraction_result(),
            story_result=story_result,
            contrarian_status=contrarian_status,
            contrarian_evidence_count=contrarian_evidence,
        )

    def _ensure_source_node(self, classification: NewsletterClassification) -> int:
        node_id = self._kg_store.get_or_create_node(
            "newsletter_source", classification.publication_name,
        )
        self._nl_store.get_or_create_source(
            node_id=node_id,
            publication_name=classification.publication_name,
            sender_pattern=classification.sender_address,
        )
        return node_id

    def _score_and_record_topics(
        self,
        *,
        intelligence: NewsletterIntelligence,
        source_node_id: int,
    ) -> Tuple[List[int], int]:
        """Score each primary topic and record attention for novel ones.

        Returns (novel_topic_ids, novel_count). Topics below NOVELTY_THRESHOLD
        are filtered out here — they never reach story threading, momentum, or
        contrarian analysis.
        """
        novel_ids: List[int] = []

        for topic_name in intelligence.primary_topics:
            topic_node_id = self._kg_store.get_or_create_node("topic", topic_name)
            novelty = compute_topic_novelty(topic_node_id, self._nl_store)

            if novelty <= NOVELTY_THRESHOLD:
                logger.info(
                    "Topic below novelty threshold; skipping downstream",
                    extra={
                        "topic": topic_name,
                        "novelty_score": round(novelty, 3),
                        "threshold": NOVELTY_THRESHOLD,
                    },
                )
                continue

            novel_ids.append(topic_node_id)
            self._nl_store.record_topic_attention(
                topic_node_id=topic_node_id,
                source_node_id=source_node_id,
                novelty_score=novelty,
            )

        return novel_ids, len(novel_ids)

    async def _run_momentum_check(self) -> None:
        try:
            accelerating = await check_and_surface_momentum(
                self._kg_store, self._nl_store,
            )
            if accelerating:
                logger.info(
                    "Momentum check surfaced signals",
                    extra={"count": len(accelerating)},
                )
        except Exception as exc:
            logger.exception("Periodic momentum check failed: %s", exc)


_processor_instance: Optional[NewsletterIntelligenceProcessor] = None


def get_newsletter_processor() -> NewsletterIntelligenceProcessor:
    global _processor_instance
    if _processor_instance is None:
        from ..store import get_knowledge_graph_store
        from .store_ext import get_newsletter_graph_store
        _processor_instance = NewsletterIntelligenceProcessor(
            kg_store=get_knowledge_graph_store(),
            nl_store=get_newsletter_graph_store(),
        )
    return _processor_instance


__all__ = [
    "NewsletterIntelligenceProcessor",
    "NewsletterProcessingResult",
    "get_newsletter_processor",
]
