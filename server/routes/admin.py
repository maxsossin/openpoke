"""Admin/dev endpoints — not intended for production exposure."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query, status

from ..services.gmail.client import (
    execute_gmail_tool_with_size_guard,
    get_active_gmail_user_id,
)
from ..services.gmail.processing import EmailTextCleaner, ProcessedEmail, parse_gmail_fetch_response
from ..services.knowledge_graph import get_knowledge_graph_store, get_knowledge_graph_watcher
from ..logging_config import logger

router = APIRouter(prefix="/admin", tags=["admin"])

_cleaner = EmailTextCleaner(max_url_length=60)

# Maximum emails fetchable in a single backfill call.
# Composio's GMAIL_FETCH_EMAILS does not paginate; bump if needed.
_BACKFILL_HARD_CAP = 200


@router.post("/backfill-newsletters")
async def backfill_newsletters(
    gmail_query: str = Query(
        default="category:promotions OR category:updates",
        description=(
            "Gmail search query. Examples:\n"
            "  category:promotions newer_than:30d\n"
            "  from:bloomberg.com OR from:ft.com newer_than:90d\n"
            "  label:newsletters"
        ),
    ),
    max_results: int = Query(
        default=50,
        ge=1,
        le=_BACKFILL_HARD_CAP,
        description="Max emails to fetch (1–200). Make multiple calls to process more.",
    ),
    force: bool = Query(
        default=False,
        description=(
            "Re-process emails that were already marked processed. "
            "Useful when the newsletter pipeline wasn't in place when they were first seen."
        ),
    ),
) -> Dict[str, Any]:
    """Backfill old newsletters through the newsletter intelligence pipeline.

    The normal watcher only looks back 10 minutes, so historical newsletters
    never get processed. This endpoint fetches emails matching gmail_query,
    optionally skips already-processed ones, and runs the full pipeline:
    classify → LLM extract → novelty gate → story threading → contrarian detection.

    Returns a summary of what was found and processed. Call multiple times with
    different date ranges to process large backlogs.
    """
    composio_user_id = get_active_gmail_user_id()
    if not composio_user_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gmail is not connected. Connect via /api/v1/gmail/connect first.",
        )

    # Fetch from Gmail
    try:
        raw = execute_gmail_tool_with_size_guard(
            "GMAIL_FETCH_EMAILS",
            composio_user_id,
            arguments={
                "query": gmail_query,
                "include_payload": True,
                "max_results": max_results,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gmail fetch failed: {exc}",
        )

    emails, _ = parse_gmail_fetch_response(raw, query=gmail_query, cleaner=_cleaner)
    fetched_count = len(emails)

    store = get_knowledge_graph_store()

    if not force:
        to_process: List[ProcessedEmail] = [
            e for e in emails if not store.is_email_processed(e.id)
        ]
    else:
        # Unmark so the pipeline re-runs cleanly (watcher marks processed after extraction)
        to_process = emails

    skipped_count = fetched_count - len(to_process)

    if not to_process:
        return {
            "fetched": fetched_count,
            "skipped_already_processed": skipped_count,
            "processed": 0,
            "message": (
                "All fetched emails were already processed. "
                "Use force=true to reprocess, or widen the gmail_query date range."
            ),
        }

    logger.info(
        "Newsletter backfill starting",
        extra={
            "query": gmail_query,
            "fetched": fetched_count,
            "to_process": len(to_process),
            "force": force,
        },
    )

    watcher = get_knowledge_graph_watcher()
    await watcher.process_batch(to_process)

    logger.info(
        "Newsletter backfill complete",
        extra={"processed": len(to_process)},
    )

    return {
        "fetched": fetched_count,
        "skipped_already_processed": skipped_count,
        "processed": len(to_process),
        "gmail_query": gmail_query,
        "hint": (
            "Check server logs for per-email details (publication, novel_topics, story, "
            "contrarian_status). Query the KG with newsletter_query modes to inspect results."
        ),
    }
