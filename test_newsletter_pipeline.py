#!/usr/bin/env python3
"""Newsletter intelligence pipeline smoke test.

Exercises all four features against a temporary SQLite database so results
are deterministic and production data is never touched.

Phases tested WITHOUT any LLM call:
    0 — Newsletter classifier (heuristic)
    1 — Novelty scoring (new topic vs saturated topic)
    2 — Story threading (1-source guard, 2-source creation, word-overlap match)
    3 — Story narrative query (framings from both publishers appear)
    4 — Contrarian DB operations (pending → confirmed lifecycle)
    5 — Momentum attention recording and top-topics query

Phase tested WITH a real LLM call (opt-in via --with-llm):
    6 — Full processor.process_email() end-to-end with realistic email text

Usage:
    python test_newsletter_pipeline.py
    python test_newsletter_pipeline.py --with-llm
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results: List[bool] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    _results.append(condition)
    return condition


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Fake email factory
# ---------------------------------------------------------------------------

def make_email(
    email_id: str,
    sender: str,
    subject: str,
    body: str,
    label_ids: List[str] | None = None,
) -> "ProcessedEmail":  # noqa: F821 – resolved at runtime
    from server.services.gmail.processing import ProcessedEmail

    return ProcessedEmail(
        id=email_id,
        thread_id=email_id,
        query="test",
        subject=subject,
        sender=sender,
        recipient="maxdavidsossin@gmail.com",
        timestamp=datetime.now(timezone.utc),
        label_ids=label_ids or ["INBOX"],
        clean_text=body,
        has_attachments=False,
        attachment_count=0,
        attachment_filenames=[],
    )


# ---------------------------------------------------------------------------
# Phase 0: classifier
# ---------------------------------------------------------------------------

def test_classifier() -> None:
    section("Phase 0 — Newsletter Classifier (no LLM)")
    from server.services.knowledge_graph.newsletter.classifier import classify_newsletter

    bloomberg = make_email(
        "bb-01",
        sender="newsletter@bloomberg.com",
        subject="Bloomberg Morning Briefing",
        body="Today's top market stories. Unsubscribe here.",
    )
    result = classify_newsletter(bloomberg)
    check("Bloomberg email → is_newsletter=True", result.is_newsletter)
    check("Bloomberg publication name resolved", result.publication_name == "Bloomberg",
          f"got '{result.publication_name}'")

    personal = make_email(
        "p-01",
        sender="alice@example.com",
        subject="Lunch tomorrow?",
        body="Hey, want to grab lunch tomorrow at noon?",
    )
    result2 = classify_newsletter(personal)
    check("Personal email → is_newsletter=False", not result2.is_newsletter)


# ---------------------------------------------------------------------------
# Phase 1: novelty scoring
# ---------------------------------------------------------------------------

def test_novelty(kg_store, nl_store) -> None:
    section("Phase 1 — Novelty Scoring")
    from server.services.knowledge_graph.newsletter.novelty import (
        NOVELTY_THRESHOLD,
        compute_topic_novelty,
    )

    # Brand-new topic — no attention records yet
    new_topic_id = kg_store.get_or_create_node("topic", "quantum computing breakthroughs")
    score = compute_topic_novelty(new_topic_id, nl_store)
    check("New topic scores 1.0 novelty", score == 1.0, f"score={score:.3f}")

    # Saturate the topic (8 mentions = _SATURATION_COUNT)
    source_id = kg_store.get_or_create_node("newsletter_source", "Test Source")
    for i in range(8):
        nl_store.record_topic_attention(
            topic_node_id=new_topic_id,
            source_node_id=source_id,
            novelty_score=1.0,
        )

    score2 = compute_topic_novelty(new_topic_id, nl_store)
    check(
        f"Saturated topic (8 mentions) scores ≤ {NOVELTY_THRESHOLD}",
        score2 <= NOVELTY_THRESHOLD,
        f"score={score2:.3f}",
    )


# ---------------------------------------------------------------------------
# Phase 2: story threading
# ---------------------------------------------------------------------------

async def test_story_threading(kg_store, nl_store) -> None:
    section("Phase 2 — Story Threading")
    from server.services.knowledge_graph.newsletter.extractor import NewsletterIntelligence
    from server.services.knowledge_graph.newsletter.story_threader import (
        STORY_MIN_SOURCES,
        thread_story,
    )

    bloomberg_id = kg_store.get_or_create_node("newsletter_source", "Bloomberg")
    ft_id = kg_store.get_or_create_node("newsletter_source", "Financial Times")
    topic_id = kg_store.get_or_create_node("topic", "Federal Reserve interest rate policy")

    intel = NewsletterIntelligence(
        primary_topics=["Federal Reserve interest rate policy"],
        story_title="Fed Rate Hike Decision",
        framing_text=(
            "Bloomberg reports the Federal Reserve is on track to raise rates by 50bps "
            "at its December meeting, citing persistent inflation above the 2% target."
        ),
        sentiment="neutral",
        key_claim="Fed will raise rates 50bps in December",
    )

    email_bb = make_email(
        "bb-02", "newsletter@bloomberg.com",
        "Fed set to hike rates", intel.framing_text,
    )

    # --- 2a: First source — story should NOT be created yet ---
    # Record attention for Bloomberg only (1 source)
    nl_store.record_topic_attention(
        topic_node_id=topic_id,
        source_node_id=bloomberg_id,
        novelty_score=1.0,
    )

    result1 = await thread_story(
        email=email_bb,
        intelligence=intel,
        source_node_id=bloomberg_id,
        topic_node_ids=[topic_id],
        kg_store=kg_store,
        nl_store=nl_store,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )
    check(
        f"1 source → story deferred (need {STORY_MIN_SOURCES})",
        result1.story_node_id is None,
        f"story_node_id={result1.story_node_id}",
    )

    # --- 2b: Second source — story SHOULD be created ---
    intel_ft = NewsletterIntelligence(
        primary_topics=["Federal Reserve interest rate policy"],
        story_title="Federal Reserve Rate Decision",  # slightly different wording — tests overlap
        framing_text=(
            "The Financial Times sees the Fed's rate decision as a pivot point for "
            "global bond markets, with European central banks watching closely."
        ),
        sentiment="alarmed",
        key_claim="Global bond markets will reprice if Fed hikes 50bps",
    )

    email_ft = make_email(
        "ft-02", "newsletter@ft.com",
        "Fed decision rattles bond markets", intel_ft.framing_text,
    )

    nl_store.record_topic_attention(
        topic_node_id=topic_id,
        source_node_id=ft_id,
        novelty_score=0.9,
    )

    result2 = await thread_story(
        email=email_ft,
        intelligence=intel_ft,
        source_node_id=ft_id,
        topic_node_ids=[topic_id],
        kg_store=kg_store,
        nl_store=nl_store,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )
    check(
        "2 sources → story node created",
        result2.story_node_id is not None,
        f"story_node_id={result2.story_node_id}, was_created={result2.was_created}",
    )
    check("Story was_created=True", result2.was_created)

    # --- 2c: Third email on same story — should match by word overlap ---
    intel_nyt = NewsletterIntelligence(
        primary_topics=["Federal Reserve interest rate policy"],
        story_title="Fed Rate Decision",  # overlaps "rate" + "decision" with "Federal Reserve Rate Decision" → 0.67 > 0.60
        framing_text=(
            "The New York Times analysis suggests the Fed rate hike will have "
            "outsized effects on mortgage markets and housing affordability."
        ),
        sentiment="alarmed",
        key_claim="Mortgage rates could rise above 8% post-hike",
    )
    nyt_id = kg_store.get_or_create_node("newsletter_source", "New York Times")
    email_nyt = make_email(
        "nyt-02", "newsletter@nytimes.com",
        "Fed rate impact on housing", intel_nyt.framing_text,
    )
    nl_store.record_topic_attention(
        topic_node_id=topic_id,
        source_node_id=nyt_id,
        novelty_score=0.7,
    )

    result3 = await thread_story(
        email=email_nyt,
        intelligence=intel_nyt,
        source_node_id=nyt_id,
        topic_node_ids=[topic_id],
        kg_store=kg_store,
        nl_store=nl_store,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )
    check(
        "3rd email threads to existing story (word overlap)",
        result3.story_node_id is not None and not result3.was_created,
        f"story_node_id={result3.story_node_id}, was_created={result3.was_created}",
    )

    return result2.story_node_id  # return story_node_id for use in later phases


# ---------------------------------------------------------------------------
# Phase 3: story narrative query
# ---------------------------------------------------------------------------

def test_story_narrative(story_node_id: int, nl_store) -> None:
    section("Phase 3 — Story Narrative Query")

    narrative = nl_store.query_story_narrative(story_node_id)
    check("Narrative returned for story node", bool(narrative))
    check(
        "At least 2 source framings recorded",
        narrative.get("source_count", 0) >= 2,
        f"source_count={narrative.get('source_count')}",
    )
    check(
        "Framings list is non-empty",
        len(narrative.get("framings", [])) >= 2,
        f"framings={len(narrative.get('framings', []))}",
    )

    sentiments = {f["sentiment"] for f in narrative.get("framings", [])}
    sources = narrative.get("sources", [])
    print(f"     sources: {sources}")
    print(f"     sentiments: {sentiments}")
    print(f"     story name: {narrative.get('story')}")


# ---------------------------------------------------------------------------
# Phase 4: contrarian DB lifecycle
# ---------------------------------------------------------------------------

def test_contrarian_db(story_node_id: int, kg_store, nl_store) -> None:
    section("Phase 4 — Contrarian DB Lifecycle (no LLM)")
    from server.services.knowledge_graph.newsletter.contrarian import (
        CONTRARIAN_MIN_DISSENTER_EVIDENCE,
    )

    economist_id = kg_store.get_or_create_node("newsletter_source", "The Economist")
    topic_id = kg_store.get_or_create_node("topic", "Federal Reserve interest rate policy")

    # First evidence email → should be pending
    status1, count1 = nl_store.upsert_contrarian_position(
        topic_node_id=topic_id,
        source_node_id=economist_id,
        position_text=(
            "The Economist argues the Fed is making a policy error: rate hikes now "
            "risk tipping the economy into recession when inflation is already cooling."
        ),
        prevailing_view="Most publications expect the rate hike to control inflation without recession.",
        email_id="economist-01",
        min_evidence_for_confirmed=CONTRARIAN_MIN_DISSENTER_EVIDENCE,
    )
    check("First evidence → status=pending", status1 == "pending",
          f"status={status1}, count={count1}")
    check("First evidence → count=1", count1 == 1)

    # Second evidence email → should become confirmed
    status2, count2 = nl_store.upsert_contrarian_position(
        topic_node_id=topic_id,
        source_node_id=economist_id,
        position_text=(
            "The Economist doubles down: the Fed's continued tightening ignores "
            "leading indicators that inflation is already retreating."
        ),
        prevailing_view="Most publications expect the rate hike to control inflation without recession.",
        email_id="economist-02",
        min_evidence_for_confirmed=CONTRARIAN_MIN_DISSENTER_EVIDENCE,
    )
    check(
        f"Second evidence → status=confirmed (threshold={CONTRARIAN_MIN_DISSENTER_EVIDENCE})",
        status2 == "confirmed",
        f"status={status2}, count={count2}",
    )
    check("Second evidence → count=2", count2 == 2)

    # Deduplication: same email_id should not increment count
    status3, count3 = nl_store.upsert_contrarian_position(
        topic_node_id=topic_id,
        source_node_id=economist_id,
        position_text="Repeated email — same ID as economist-02",
        prevailing_view="Same prevailing view",
        email_id="economist-02",  # duplicate
        min_evidence_for_confirmed=CONTRARIAN_MIN_DISSENTER_EVIDENCE,
    )
    check("Duplicate email_id → count unchanged", count3 == 2,
          f"count={count3} (should still be 2)")

    confirmed = nl_store.get_confirmed_contrarians(limit=20)
    check("get_confirmed_contrarians returns ≥1 result", len(confirmed) >= 1,
          f"count={len(confirmed)}")


# ---------------------------------------------------------------------------
# Phase 5: momentum tracking
# ---------------------------------------------------------------------------

def test_momentum(kg_store, nl_store) -> None:
    section("Phase 5 — Momentum Tracking")

    topic_a = kg_store.get_or_create_node("topic", "US AI regulation")
    src1 = kg_store.get_or_create_node("newsletter_source", "Axios")
    src2 = kg_store.get_or_create_node("newsletter_source", "Wired")

    # Simulate 3 mentions from 2 sources this week
    for _ in range(2):
        nl_store.record_topic_attention(topic_node_id=topic_a, source_node_id=src1, novelty_score=0.9)
    nl_store.record_topic_attention(topic_node_id=topic_a, source_node_id=src2, novelty_score=0.8)

    stats = nl_store.get_topic_window_stats(topic_a, recent_days=7, prior_days=7)
    check(
        "Recent mention count ≥ 3",
        stats["recent_mentions"] >= 3,
        f"recent_mentions={stats['recent_mentions']}",
    )
    check(
        "Recent source count = 2",
        stats["recent_sources"] == 2,
        f"recent_sources={stats['recent_sources']}",
    )

    top = nl_store.get_top_momentum_topics(limit=5)
    topic_names = [t["canonical_name"] for t in top]
    check(
        "get_top_momentum_topics includes the topic",
        "US AI regulation" in topic_names,
        f"topics={topic_names}",
    )


# ---------------------------------------------------------------------------
# Phase 6: full processor with real LLM (opt-in)
# ---------------------------------------------------------------------------

async def test_full_processor_with_llm(kg_store, nl_store) -> None:
    section("Phase 6 — Full Processor + Real LLM")
    print("  (makes real API calls; results are non-deterministic)")

    from server.services.knowledge_graph.newsletter.processor import NewsletterIntelligenceProcessor

    processor = NewsletterIntelligenceProcessor(kg_store=kg_store, nl_store=nl_store)

    emails = [
        make_email(
            "bb-llm-01",
            sender="newsletter@bloomberg.com",
            subject="Bloomberg: Fed raises rates 50bps",
            body=(
                "The Federal Reserve raised interest rates by 50 basis points today, "
                "its most aggressive move since 2000. Fed Chair Jerome Powell signaled "
                "more hikes are likely as inflation remains above the 2% target. "
                "Markets fell sharply on the news, with the S&P 500 dropping 2.3%. "
                "Unsubscribe from Bloomberg newsletters here."
            ),
            label_ids=["INBOX", "CATEGORY_UPDATES"],
        ),
        make_email(
            "ft-llm-01",
            sender="newsletter@ft.com",
            subject="FT: Federal Reserve moves shake bond markets",
            body=(
                "The Federal Reserve's decision to raise rates by 50 basis points "
                "sent shockwaves through global bond markets. The 10-year Treasury "
                "yield spiked to 4.8%, its highest since 2007. The Financial Times "
                "analysis suggests this signals a structural shift in monetary policy "
                "that could persist through 2025. Unsubscribe here."
            ),
            label_ids=["INBOX", "CATEGORY_UPDATES"],
        ),
    ]

    results = []
    for email in emails:
        result = await processor.process_email(email)
        results.append(result)
        if result:
            print(f"     [{email.id}] publication={result.source_name}, "
                  f"novel_topics={result.novel_topic_count}, "
                  f"story={result.story_result.story_name if result.story_result else None}, "
                  f"contrarian={result.contrarian_status}")
        else:
            print(f"     [{email.id}] not classified as newsletter")

    classified = [r for r in results if r is not None and r.is_newsletter]
    check(
        "Both emails classified as newsletters",
        len(classified) == 2,
        f"{len(classified)}/2 classified",
    )
    with_topics = [r for r in classified if r.novel_topic_count > 0]
    check(
        "At least one email produced novel topics",
        len(with_topics) >= 1,
        f"{len(with_topics)} emails with novel topics",
    )
    with_story = [r for r in classified if r.story_result and r.story_result.story_node_id]
    print(f"     story nodes created/matched: {len(with_story)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(with_llm: bool) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_newsletter.db"

        print(f"\nUsing temp DB: {db_path}")

        from server.services.knowledge_graph.store import KnowledgeGraphStore
        from server.services.knowledge_graph.newsletter.store_ext import NewsletterGraphStore

        kg_store = KnowledgeGraphStore(db_path=db_path)
        nl_store = NewsletterGraphStore(kg_store)

        test_classifier()
        test_novelty(kg_store, nl_store)
        story_node_id = await test_story_threading(kg_store, nl_store)
        if story_node_id:
            test_story_narrative(story_node_id, nl_store)
        else:
            print("\n  [SKIP] Phase 3 — story_node_id is None, threading test failed")
        test_contrarian_db(story_node_id or 1, kg_store, nl_store)
        test_momentum(kg_store, nl_store)

        if with_llm:
            await test_full_processor_with_llm(kg_store, nl_store)
        else:
            print(f"\n{'─' * 60}")
            print("  Phase 6 — Full Processor + Real LLM: SKIPPED")
            print("  Run with --with-llm to include this phase")

    # Summary
    total = len(_results)
    passed = sum(_results)
    failed = total - passed
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  (\033[31m{failed} failed\033[0m)")
    else:
        print(f"  (\033[32mall passed\033[0m)")
    print(f"{'=' * 60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    with_llm = "--with-llm" in sys.argv
    asyncio.run(main(with_llm=with_llm))
