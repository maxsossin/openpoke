"""Background task that extracts knowledge graph facts from incoming emails."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from ..gmail.client import execute_gmail_tool_with_size_guard as execute_gmail_tool, get_active_gmail_user_id
from ..gmail.processing import EmailTextCleaner, ProcessedEmail, parse_gmail_fetch_response
from .extractor import ExtractionResult, extract_from_email
from .store import KnowledgeGraphStore, get_knowledge_graph_store
from .newsletter.processor import NewsletterProcessingResult, get_newsletter_processor
from ...logging_config import logger

_DEFAULT_POLL_INTERVAL_SECONDS = 60.0
_DEFAULT_LOOKBACK_MINUTES = 10
_DEFAULT_MAX_RESULTS = 10

# Run stale-fact deprecation once per hour at the default poll interval
_DEPRECATION_INTERVAL_POLLS = 60


class KnowledgeGraphWatcher:
    """Background asyncio task that extracts knowledge graph facts from Gmail.

    Architecture mirrors ImportantEmailWatcher: independent asyncio.Task,
    same Gmail query, separate deduplication (kg_processed_emails table
    instead of gmail_seen.json). The two watchers are fully decoupled —
    neither one depends on the other's seen-state or scheduling.

    First poll seeds all current inbox emails as processed without extracting,
    preventing a burst of LLM calls on startup.
    """

    def __init__(
        self,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        lookback_minutes: int = _DEFAULT_LOOKBACK_MINUTES,
        *,
        store: Optional[KnowledgeGraphStore] = None,
    ) -> None:
        self._poll_interval = poll_interval_seconds
        self._lookback_minutes = lookback_minutes
        self._store = store or get_knowledge_graph_store()
        self._cleaner = EmailTextCleaner(max_url_length=60)
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._has_seeded_initial_snapshot = False
        self._poll_count = 0

    async def start(self) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                return
            loop = asyncio.get_running_loop()
            self._running = True
            self._has_seeded_initial_snapshot = False
            self._poll_count = 0
            self._task = loop.create_task(self._run(), name="knowledge-graph-watcher")
            logger.info(
                "Knowledge graph watcher started",
                extra={
                    "interval_seconds": self._poll_interval,
                    "lookback_minutes": self._lookback_minutes,
                },
            )

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._task = None
                logger.info("Knowledge graph watcher stopped")

    async def _run(self) -> None:
        try:
            while self._running:
                try:
                    await self._poll_once()
                except Exception as exc:
                    logger.exception(
                        "Knowledge graph watcher poll failed",
                        extra={"error": str(exc)},
                    )
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            raise

    async def _poll_once(self) -> None:
        self._poll_count += 1

        composio_user_id = get_active_gmail_user_id()
        if not composio_user_id:
            logger.debug("Gmail not connected; skipping knowledge graph poll")
            return

        query = f"label:INBOX newer_than:{self._lookback_minutes}m"
        arguments = {
            "query": query,
            "include_payload": True,
            "max_results": _DEFAULT_MAX_RESULTS,
        }

        try:
            raw_result = execute_gmail_tool(
                "GMAIL_FETCH_EMAILS", composio_user_id, arguments=arguments
            )
        except Exception as exc:
            logger.warning(
                "Failed to fetch Gmail messages for KG watcher",
                extra={"error": str(exc)},
            )
            return

        processed_emails, _ = parse_gmail_fetch_response(
            raw_result, query=query, cleaner=self._cleaner
        )

        if not self._has_seeded_initial_snapshot:
            for email in processed_emails:
                self._store.mark_email_processed(email.id)
            logger.info(
                "Knowledge graph watcher seeded initial snapshot",
                extra={"seeded_count": len(processed_emails)},
            )
            self._has_seeded_initial_snapshot = True
            return

        unprocessed: List[ProcessedEmail] = [
            email for email in processed_emails
            if not self._store.is_email_processed(email.id)
        ]

        if not unprocessed:
            logger.debug("Knowledge graph watcher: no new emails to process")
        else:
            await self._extract_batch(unprocessed)

        if self._poll_count % _DEPRECATION_INTERVAL_POLLS == 0:
            try:
                deprecated = self._store.deprecate_stale_facts()
                logger.debug(
                    "Knowledge graph periodic deprecation ran",
                    extra={"deprecated": deprecated},
                )
            except Exception as exc:
                logger.warning(
                    "Knowledge graph deprecation check failed",
                    extra={"error": str(exc)},
                )

    async def _extract_batch(self, emails: List[ProcessedEmail]) -> None:
        """Extract knowledge graph facts from a batch of unprocessed emails.

        Newsletter emails are routed through the newsletter intelligence
        processor, which: classifies the source, runs a combined LLM
        extraction, scores topic novelty, threads story nodes, and detects
        contrarian positions. The processor returns an ExtractionResult that
        is then fed into the standard _apply_extraction() path so all KG
        writes remain consistent. Non-newsletter emails follow the original
        extract_from_email() → _apply_extraction() path unchanged.
        """
        processor = get_newsletter_processor()
        extracted_count = 0

        for email in emails:
            try:
                # Newsletter path: combined extraction + intelligence pipeline
                nl_result: NewsletterProcessingResult | None = (
                    await processor.process_email(email)
                )

                if nl_result is not None:
                    # Newsletter: use extraction result returned by processor
                    if nl_result.extraction_result:
                        self._apply_extraction(email, nl_result.extraction_result)
                        extracted_count += 1
                    else:
                        # Newsletter LLM failed; fall back to standard extraction
                        fallback = await extract_from_email(email)
                        if fallback:
                            self._apply_extraction(email, fallback)
                            extracted_count += 1
                    logger.info(
                        "Newsletter processed",
                        extra={
                            "email_id": email.id,
                            "publication": nl_result.source_name,
                            "novel_topics": nl_result.novel_topic_count,
                            "story": (
                                nl_result.story_result.story_name
                                if nl_result.story_result
                                else None
                            ),
                            "contrarian_status": nl_result.contrarian_status,
                        },
                    )
                else:
                    # Standard path: unchanged
                    result = await extract_from_email(email)
                    if result:
                        self._apply_extraction(email, result)
                        extracted_count += 1

            except Exception as exc:
                logger.warning(
                    "Failed to extract KG facts from email",
                    extra={"email_id": email.id, "error": str(exc)},
                )
            finally:
                # Mark processed regardless of success to prevent infinite retry
                self._store.mark_email_processed(email.id)

        logger.info(
            "Knowledge graph extraction batch complete",
            extra={
                "attempted": len(emails),
                "with_extractions": extracted_count,
            },
        )

    def _apply_extraction(self, email: ProcessedEmail, result: ExtractionResult) -> None:
        """Write extracted entities and relationships into the graph store."""
        extracted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        source_email_timestamp = email.timestamp.isoformat(timespec="seconds")

        # Build name→node_id map for edge resolution within this extraction
        name_to_node_id: dict[str, int] = {}

        for entity in result.entities:
            try:
                node_id = self._store.get_or_create_node(
                    entity.node_type, entity.canonical_name
                )
                name_to_node_id[entity.canonical_name] = node_id

                for fact in entity.facts:
                    inserted, contradiction = self._store.upsert_fact(
                        node_id=node_id,
                        fact_key=fact.key,
                        fact_value=fact.value,
                        confidence=fact.confidence,
                        source_email_id=email.id,
                        source_email_timestamp=source_email_timestamp,
                        extracted_at=extracted_at,
                    )
                    if contradiction:
                        logger.info(
                            "Knowledge graph contradiction detected and versioned",
                            extra={
                                "node_id": node_id,
                                "canonical_name": entity.canonical_name,
                                "fact_key": fact.key,
                                "new_value": fact.value,
                                "source_email_id": email.id,
                            },
                        )
            except Exception as exc:
                logger.warning(
                    "Failed to write entity to knowledge graph",
                    extra={
                        "entity": entity.canonical_name,
                        "email_id": email.id,
                        "error": str(exc),
                    },
                )

        for rel in result.relationships:
            from_id = name_to_node_id.get(rel.from_entity)
            to_id = name_to_node_id.get(rel.to_entity)
            if from_id is None or to_id is None:
                logger.debug(
                    "Skipping edge: one or both entities not found in current extraction",
                    extra={"from": rel.from_entity, "to": rel.to_entity},
                )
                continue
            try:
                self._store.upsert_edge(
                    from_node_id=from_id,
                    to_node_id=to_id,
                    edge_type=rel.edge_type,
                    confidence=rel.confidence,
                    source_email_id=email.id,
                    source_email_timestamp=source_email_timestamp,
                    extracted_at=extracted_at,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to write edge to knowledge graph",
                    extra={"error": str(exc)},
                )


    async def process_batch(self, emails: List[ProcessedEmail]) -> None:
        """Public entry point for external callers (e.g. backfill tooling).

        Runs the same classification → extraction → KG-write pipeline as the
        normal poll loop, then marks each email processed so it won't be
        re-attempted by future polls.
        """
        await self._extract_batch(emails)


_watcher_instance: Optional[KnowledgeGraphWatcher] = None


def get_knowledge_graph_watcher() -> KnowledgeGraphWatcher:
    global _watcher_instance
    if _watcher_instance is None:
        _watcher_instance = KnowledgeGraphWatcher()
    return _watcher_instance


__all__ = ["KnowledgeGraphWatcher", "get_knowledge_graph_watcher"]
