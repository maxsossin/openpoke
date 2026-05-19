"""Newsletter-specific LLM extraction.

For newsletter emails this replaces the standard extract_from_email() call.
A single LLM invocation returns both the standard knowledge graph entities
(persons, organisations, topics, events) AND newsletter-specific intelligence
(framing, sentiment, story title, key claim). One call, not two.

The ExtractionResult produced here is fed into the existing _apply_extraction()
path unchanged — standard KG writes are not bypassed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...gmail.processing import ProcessedEmail
from ....config import get_settings
from ....logging_config import logger
from ....openrouter_client import request_chat_completion
from ..extractor import (
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    ExtractionResult,
)

_TOOL_NAME = "extract_newsletter_intelligence"

_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": (
            "Extract structured intelligence from a newsletter email. "
            "Returns standard knowledge graph entities AND newsletter-specific framing data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "description": "Named entities: persons, organisations, topics, events.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_type": {
                                "type": "string",
                                "enum": ["person", "organization", "topic", "event"],
                            },
                            "canonical_name": {
                                "type": "string",
                                "description": (
                                    "Stable label: full name for persons, official name for "
                                    "organisations, concise phrase for topics/events."
                                ),
                            },
                            "facts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "key": {"type": "string"},
                                        "value": {"type": "string"},
                                        "confidence": {"type": "number"},
                                    },
                                    "required": ["key", "value", "confidence"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["node_type", "canonical_name", "facts"],
                        "additionalProperties": False,
                    },
                },
                "relationships": {
                    "type": "array",
                    "description": "Directed relationships between extracted entities.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_entity": {"type": "string"},
                            "to_entity": {"type": "string"},
                            "edge_type": {
                                "type": "string",
                                "enum": [
                                    "works_at", "sent_to", "involved_in",
                                    "mentions", "knows",
                                ],
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "from_entity", "to_entity", "edge_type", "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "primary_topics": {
                    "type": "array",
                    "description": (
                        "Canonical names of the 1–3 primary topics this newsletter issue "
                        "addresses. Must match canonical_name values in entities. "
                        "Empty array when the issue has no clear focus."
                    ),
                    "items": {"type": "string"},
                },
                "story_title": {
                    "type": "string",
                    "description": (
                        "A concise title (≤8 words) for the lead or most prominent ongoing "
                        "story in this issue. Even if the newsletter covers multiple topics, "
                        "identify the one the publication leads with or devotes the most "
                        "attention to. Use a stable, canonical phrasing (e.g. 'Fed Rate "
                        "Decision', 'US-China Trade War') so the same story matches across "
                        "different publications. Empty string only when the email is purely "
                        "transactional (job alerts, receipts, account notifications)."
                    ),
                },
                "framing_text": {
                    "type": "string",
                    "description": (
                        "2–3 sentences capturing this publication's specific angle, emphasis, "
                        "or position on the primary topic. Include any specific prediction or "
                        "claim the publication is making. This must faithfully represent the "
                        "publication's actual viewpoint — not a neutral synthesis. "
                        "If Bloomberg is bearish while others are bullish, say so explicitly. "
                        "Always provide framing_text when story_title is non-empty."
                    ),
                },
                "sentiment": {
                    "type": "string",
                    "enum": [
                        "bullish", "bearish", "neutral",
                        "alarmed", "skeptical", "optimistic",
                    ],
                    "description": "Publication's overall tone toward the primary topic.",
                },
                "key_claim": {
                    "type": "string",
                    "description": (
                        "The single most specific, falsifiable claim or prediction in this issue. "
                        "Examples: 'Fed will cut rates by March 2025', "
                        "'Apple market cap will exceed $4T by year-end'. "
                        "Empty string when no specific claim is present."
                    ),
                },
            },
            "required": [
                "entities", "relationships", "primary_topics",
                "story_title", "framing_text", "sentiment", "key_claim",
            ],
            "additionalProperties": False,
        },
    },
}

_SYSTEM_PROMPT = (
    "You extract structured intelligence from newsletter emails for a private knowledge graph. "
    "Extract: (1) standard entities — people with roles and organisations, topics, events; "
    "(2) this publication's specific framing and position on its primary topics, not a neutral "
    "summary. Preserve the publication's actual angle faithfully. If the publication is "
    "pessimistic about a policy while others are optimistic, that divergence is the signal. "
    "Use confidence 0.8–1.0 for explicit facts, 0.5–0.7 for clearly implied. "
    "canonical_name must be stable across emails from the same or different sources."
)

# Cap newsletter body sent to LLM to control token cost
_MAX_BODY_CHARS = 8_000


@dataclass
class NewsletterIntelligence:
    """Combined extraction result for newsletter emails."""

    primary_topics: List[str]
    story_title: str
    framing_text: str
    sentiment: str
    key_claim: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)

    @property
    def has_story(self) -> bool:
        return bool(self.story_title.strip())

    @property
    def has_framing(self) -> bool:
        return bool(self.framing_text.strip())

    def to_extraction_result(self) -> ExtractionResult:
        """Convert to the standard ExtractionResult for _apply_extraction()."""
        return ExtractionResult(
            entities=self.entities,
            relationships=self.relationships,
        )


async def extract_newsletter_intelligence(
    email: ProcessedEmail,
) -> Optional[NewsletterIntelligence]:
    """Single LLM call: standard KG entities + newsletter-specific framing.

    Returns None when the API is unavailable or the response is unusable.
    The caller marks the email processed regardless of the return value.
    """
    settings = get_settings()
    api_key = settings.openrouter_api_key
    model = settings.knowledge_graph_model

    if not api_key:
        logger.warning("Newsletter extraction skipped; API key missing")
        return None

    messages = [{"role": "user", "content": _format_payload(email)}]

    try:
        response = await request_chat_completion(
            model=model,
            messages=messages,
            system=_SYSTEM_PROMPT,
            api_key=api_key,
            tools=[_TOOL_SCHEMA],
        )
    except Exception as exc:
        logger.exception(
            "Newsletter extraction API call failed [email=%s]: %s",
            email.id,
            exc,
        )
        return None

    choice = (response.get("choices") or [{}])[0]
    tool_calls = (choice.get("message") or {}).get("tool_calls") or []

    for tc in tool_calls:
        fn = tc.get("function") or {}
        if fn.get("name") != _TOOL_NAME:
            continue
        args = _coerce_args(fn.get("arguments"))
        if args is None:
            logger.warning(
                "Newsletter extraction: unparseable tool arguments",
                extra={"email_id": email.id},
            )
            return None
        return _parse(args)

    logger.debug(
        "Newsletter extraction: no tool call returned",
        extra={"email_id": email.id},
    )
    return None


def _format_payload(email: ProcessedEmail) -> str:
    return "\n".join([
        f"Publication: {email.sender}",
        f"Subject: {email.subject}",
        f"Date: {email.timestamp.isoformat()}",
        "",
        (email.clean_text or "(empty body)")[:_MAX_BODY_CHARS],
    ])


def _coerce_args(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return None


def _parse(args: Dict[str, Any]) -> NewsletterIntelligence:
    entities: List[ExtractedEntity] = []
    for raw in args.get("entities") or []:
        if not isinstance(raw, dict):
            continue
        ntype = (raw.get("node_type") or "").strip()
        cname = (raw.get("canonical_name") or "").strip()
        if not ntype or not cname:
            continue
        facts: List[ExtractedFact] = []
        for rf in raw.get("facts") or []:
            if not isinstance(rf, dict):
                continue
            k = (rf.get("key") or "").strip()
            v = (rf.get("value") or "").strip()
            if not k or not v:
                continue
            try:
                conf = float(rf.get("confidence", 1.0))
            except (TypeError, ValueError):
                conf = 1.0
            facts.append(ExtractedFact(
                key=k, value=v, confidence=max(0.0, min(1.0, conf)),
            ))
        entities.append(ExtractedEntity(
            node_type=ntype, canonical_name=cname, facts=facts,
        ))

    relationships: List[ExtractedRelationship] = []
    for raw in args.get("relationships") or []:
        if not isinstance(raw, dict):
            continue
        fe = (raw.get("from_entity") or "").strip()
        te = (raw.get("to_entity") or "").strip()
        et = (raw.get("edge_type") or "").strip()
        if not fe or not te or not et:
            continue
        try:
            conf = float(raw.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        relationships.append(ExtractedRelationship(
            from_entity=fe, to_entity=te, edge_type=et,
            confidence=max(0.0, min(1.0, conf)),
        ))

    primary_topics = [
        t.strip()
        for t in (args.get("primary_topics") or [])
        if isinstance(t, str) and t.strip()
    ]

    return NewsletterIntelligence(
        primary_topics=primary_topics,
        story_title=(args.get("story_title") or "").strip(),
        framing_text=(args.get("framing_text") or "").strip(),
        sentiment=(args.get("sentiment") or "neutral").strip(),
        key_claim=(args.get("key_claim") or "").strip(),
        entities=entities,
        relationships=relationships,
    )


__all__ = ["NewsletterIntelligence", "extract_newsletter_intelligence"]
