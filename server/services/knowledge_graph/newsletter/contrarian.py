"""Contrarian position detection.

Identifies when a newsletter source takes a position that materially
contradicts the prevailing view established by other sources covering
the same story. Dissent is modeled explicitly as a contrarian record and
surfaced at query time — it is not buried in volume or collapsed into consensus.

Two-stage evidence requirement:
    CONTRARIAN_MIN_PREVAILING_SOURCES distinct sources must have framed
    the same story before a "prevailing view" is considered established.
    This prevents a fringe source from being labelled contrarian against
    a non-existent consensus.

    CONTRARIAN_MIN_DISSENTER_EVIDENCE (2) separate emails from the dissenting
    source must confirm the position before its status advances from 'pending'
    to 'confirmed'. A single newsletter is noise; repeated dissent is a signal.

The LLM determination is conservative by design: superficial differences in
tone, emphasis, or alarm level do NOT qualify. Only specific, falsifiable
factual contradictions or directly opposed directional predictions count.

Status lifecycle: 'pending' → 'confirmed' (evidence threshold met).
Confirmed contrarians are surfaced at query time via the query interface.
They are NOT proactively dispatched — only momentum signals are proactively
surfaced. Contrarian positions are available for pull queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .store_ext import NewsletterGraphStore
from .extractor import NewsletterIntelligence
from ....config import get_settings
from ....logging_config import logger
from ....openrouter_client import request_chat_completion
from ...gmail.processing import ProcessedEmail

# Distinct sources required to establish a prevailing view.
# Entity synonym resolution (Fix 1) consolidates fragmented story nodes so
# multi-source framing counts are reliably reachable at 3.
CONTRARIAN_MIN_PREVAILING_SOURCES = 3

# Separate emails from the dissenting source required before 'confirmed'
CONTRARIAN_MIN_DISSENTER_EVIDENCE = 2

_TOOL_NAME = "assess_contrarian_position"

_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": (
            "Determine whether a new publication's framing materially contradicts "
            "the prevailing view established by other sources on the same story."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "is_contrarian": {
                    "type": "boolean",
                    "description": (
                        "True only when the new framing makes a specific claim or directional "
                        "prediction that directly contradicts the majority of other sources. "
                        "Differences in tone, alarm level, or emphasis do NOT qualify. "
                        "The contradiction must concern a specific observable fact or outcome."
                    ),
                },
                "contradiction_description": {
                    "type": "string",
                    "description": (
                        "Required when is_contrarian is true. One sentence naming the "
                        "specific point of contradiction. "
                        "Example: 'This source predicts rate cuts by March; three other "
                        "sources predict rates held steady through Q2.'"
                    ),
                },
                "prevailing_view_summary": {
                    "type": "string",
                    "description": (
                        "One sentence summarising the majority position. "
                        "Include when is_contrarian is true."
                    ),
                },
            },
            "required": [
                "is_contrarian",
                "contradiction_description",
                "prevailing_view_summary",
            ],
            "additionalProperties": False,
        },
    },
}

# The system prompt is the primary calibration instrument.
# "When in doubt, return is_contrarian=false" encodes the conservative stance.
_SYSTEM_PROMPT = (
    "You assess whether a publication is taking a contrarian position relative to consensus. "
    "You receive: (1) a new framing from one source, and (2) recent framings from multiple "
    "other sources covering the same story.\n\n"
    "Mark is_contrarian=true ONLY when the new framing makes a specific, falsifiable claim "
    "that DIRECTLY CONTRADICTS a factual claim or directional prediction made by the majority "
    "of the other sources. Examples of qualifying contradictions:\n"
    "  - New source: 'Fed will cut rates by March.' Others: 'Fed will hold through Q2.'\n"
    "  - New source: 'Company X will miss earnings.' Others: 'Company X will beat estimates.'\n\n"
    "Examples that do NOT qualify:\n"
    "  - New source is alarmed; others are calm. (Tone difference only.)\n"
    "  - New source provides more detail on the same underlying fact.\n"
    "  - New source focuses on a different aspect of the same story.\n\n"
    "When in doubt, return is_contrarian=false. False negatives are acceptable; "
    "false positives erode the user's trust in the signal."
)


@dataclass
class ContrarianAssessment:
    is_contrarian: bool
    contradiction_description: str
    prevailing_view_summary: str


async def detect_contrarian_position(
    *,
    story_node_id: int,
    source_node_id: int,
    intelligence: NewsletterIntelligence,
    email: ProcessedEmail,
    nl_store: NewsletterGraphStore,
) -> Optional[Tuple[str, int]]:
    """Check whether this newsletter's framing contradicts the prevailing view.

    Returns (status, evidence_count) if a contrarian record was created or
    updated, or None if no contrarian signal was found or evidence thresholds
    were not met.

    Guard 1: CONTRARIAN_MIN_PREVAILING_SOURCES distinct sources must have
             framed this story in the last 14 days.
    Guard 2: A single contradicting email creates a 'pending' record only.
             CONTRARIAN_MIN_DISSENTER_EVIDENCE emails advance it to 'confirmed'.
    """
    if not intelligence.has_framing:
        return None

    # Guard 1: verify prevailing view is established
    distinct_sources = nl_store.count_distinct_prevailing_sources(
        story_node_id, window_days=14,
    )
    if distinct_sources < CONTRARIAN_MIN_PREVAILING_SOURCES:
        logger.debug(
            "Contrarian check skipped: prevailing view not yet established",
            extra={
                "story_node_id": story_node_id,
                "sources": distinct_sources,
                "required": CONTRARIAN_MIN_PREVAILING_SOURCES,
            },
        )
        return None

    # Collect prevailing framings from other sources, capped to 30 days so
    # months-old framings cannot dominate the prevailing view assessment.
    prevailing = nl_store.get_story_framings(
        story_node_id,
        exclude_source_id=source_node_id,
        limit=10,
        since_days=30,
    )
    distinct_prevailing_sources = len({f["source_node_id"] for f in prevailing})
    if distinct_prevailing_sources < CONTRARIAN_MIN_PREVAILING_SOURCES:
        return None

    # LLM assessment — include key claims for direct claim-vs-claim comparison
    assessment = await _assess_via_llm(
        new_framing=intelligence.framing_text,
        new_sentiment=intelligence.sentiment,
        prevailing_framings=prevailing,
        new_key_claim=intelligence.key_claim,
    )

    if assessment is None or not assessment.is_contrarian:
        return None

    # Upsert contrarian record
    status, count = nl_store.upsert_contrarian_position(
        story_node_id=story_node_id,
        source_node_id=source_node_id,
        position_text=intelligence.framing_text,
        prevailing_view=assessment.prevailing_view_summary,
        email_id=email.id,
        min_evidence_for_confirmed=CONTRARIAN_MIN_DISSENTER_EVIDENCE,
    )

    logger.info(
        "Contrarian position updated",
        extra={
            "story_node_id": story_node_id,
            "source_node_id": source_node_id,
            "status": status,
            "evidence_count": count,
            "contradiction": assessment.contradiction_description[:120],
        },
    )

    return status, count


async def _assess_via_llm(
    new_framing: str,
    new_sentiment: str,
    prevailing_framings: List[Dict[str, Any]],
    new_key_claim: str = "",
) -> Optional[ContrarianAssessment]:
    """Single LLM call to assess whether new_framing contradicts prevailing_framings.

    new_key_claim: the most specific, falsifiable claim from the new framing.
    When present, it is included verbatim alongside prevailing key_claims so the
    LLM can perform direct claim-vs-claim comparison — the strongest signal of
    a genuine factual contradiction.
    """
    settings = get_settings()
    api_key = settings.openrouter_api_key
    if not api_key:
        return None

    prevailing_lines = []
    for f in prevailing_framings[:8]:
        line = f"  [{f['source_name']} / {f['sentiment']}]: {f['framing_text']}"
        claim = (f.get("key_claim") or "").strip()
        if claim:
            line += f" [CLAIM: {claim}]"
        prevailing_lines.append(line)
    prevailing_text = "\n".join(prevailing_lines)

    claim_line = f"\nNEW CLAIM: {new_key_claim.strip()}\n" if new_key_claim.strip() else ""
    user_content = (
        f"NEW FRAMING (sentiment: {new_sentiment}):\n{new_framing}"
        f"{claim_line}\n"
        f"PREVAILING FRAMINGS FROM OTHER SOURCES:\n{prevailing_text}"
    )

    try:
        response = await request_chat_completion(
            model=settings.knowledge_graph_model,
            messages=[{"role": "user", "content": user_content}],
            system=_SYSTEM_PROMPT,
            api_key=api_key,
            tools=[_TOOL_SCHEMA],
        )
    except Exception as exc:
        logger.exception("Contrarian LLM assessment failed: %s", exc)
        return None

    choice = (response.get("choices") or [{}])[0]
    tool_calls = (choice.get("message") or {}).get("tool_calls") or []

    for tc in tool_calls:
        fn = tc.get("function") or {}
        if fn.get("name") != _TOOL_NAME:
            continue
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                args: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw, dict):
            args = raw
        else:
            continue

        return ContrarianAssessment(
            is_contrarian=bool(args.get("is_contrarian", False)),
            contradiction_description=(
                args.get("contradiction_description") or ""
            ).strip(),
            prevailing_view_summary=(
                args.get("prevailing_view_summary") or ""
            ).strip(),
        )

    return None


__all__ = [
    "ContrarianAssessment",
    "CONTRARIAN_MIN_PREVAILING_SOURCES",
    "CONTRARIAN_MIN_DISSENTER_EVIDENCE",
    "detect_contrarian_position",
]
