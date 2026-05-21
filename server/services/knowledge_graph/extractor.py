"""LLM-powered extraction of graph entities and relationships from ProcessedEmail objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..gmail.processing import ProcessedEmail
from ...config import get_settings
from ...logging_config import logger
from ...openrouter_client import request_chat_completion

_EXTRACT_TOOL_NAME = "extract_knowledge"

_EXTRACT_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _EXTRACT_TOOL_NAME,
        "description": (
            "Extract named entities and typed relationships from an email "
            "into a structured knowledge graph."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "description": "Named entities present in the email.",
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
                                    "Stable, normalized identity: full name for persons, "
                                    "official name for organizations, concise label for "
                                    "topics/events."
                                ),
                            },
                            "facts": {
                                "type": "array",
                                "description": (
                                    "Key-value properties. Valid keys by type — "
                                    "person: email_address, role, company, relationship_to_user; "
                                    "organization: domain, industry; "
                                    "topic: description, status; "
                                    "event: date, location, participants."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "key": {"type": "string"},
                                        "value": {"type": "string"},
                                        "confidence": {
                                            "type": "number",
                                            "description": (
                                                "0.8–1.0 for explicit facts; "
                                                "0.5–0.7 for clearly implied; "
                                                "below 0.5 only if very uncertain."
                                            ),
                                        },
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
                            "from_entity": {
                                "type": "string",
                                "description": "canonical_name of the source entity.",
                            },
                            "to_entity": {
                                "type": "string",
                                "description": "canonical_name of the target entity.",
                            },
                            "edge_type": {
                                "type": "string",
                                "enum": [
                                    "works_at",
                                    "sent_to",
                                    "involved_in",
                                    "mentions",
                                    "knows",
                                    "precedes",
                                    "caused_by",
                                    "contradicts_story",
                                    "follows_from",
                                ],
                            },
                            "confidence": {"type": "number"},
                            "edge_properties": {
                                "type": "object",
                                "description": (
                                    "Optional semantic properties on the edge. "
                                    "For works_at: {\"role\": \"CTO\", \"since\": \"2022\"}. "
                                    "For involved_in: {\"capacity\": \"lead investor\"}. "
                                    "Omit when no meaningful properties exist."
                                ),
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["from_entity", "to_entity", "edge_type", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["entities", "relationships"],
            "additionalProperties": False,
        },
    },
}

_SYSTEM_PROMPT = (
    "You extract structured knowledge from email messages for a private knowledge graph. "
    "Identify people (with email addresses, roles, and organizations), organizations, "
    "topics/projects, and scheduled events. "
    "Only emit facts that are clearly stated or strongly implied by the email content. "
    "Use confidence 0.8–1.0 for explicit facts, 0.5–0.7 for clearly implied facts, "
    "and below 0.5 only when very uncertain. "
    "canonical_name must be stable across emails: use a person's full name, "
    "an organization's official name, a concise topic label. "
    "Never invent or hallucinate facts. If the email contains no extractable entities, "
    "return empty arrays."
)


@dataclass
class ExtractedFact:
    key: str
    value: str
    confidence: float


@dataclass
class ExtractedEntity:
    node_type: str
    canonical_name: str
    facts: List[ExtractedFact] = field(default_factory=list)


@dataclass
class ExtractedRelationship:
    from_entity: str
    to_entity: str
    edge_type: str
    confidence: float
    edge_properties: Optional[Dict[str, Any]] = None


@dataclass
class ExtractionResult:
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)


def _format_email_for_extraction(email: ProcessedEmail) -> str:
    lines = [
        f"From: {email.sender}",
        f"To: {email.recipient}",
        f"Subject: {email.subject}",
        f"Date: {email.timestamp.isoformat()}",
        "",
        email.clean_text or "(empty body)",
    ]
    return "\n".join(lines)


async def extract_from_email(email: ProcessedEmail) -> Optional[ExtractionResult]:
    """Call the LLM to extract a structured ExtractionResult from a single ProcessedEmail.

    Returns None when the API is unavailable or produces no usable output.
    The caller is responsible for marking the email processed regardless of result.
    """
    settings = get_settings()
    api_key = settings.openrouter_api_key
    model = settings.knowledge_graph_model

    if not api_key:
        logger.warning("Skipping KG extraction; OpenRouter API key missing")
        return None

    messages = [{"role": "user", "content": _format_email_for_extraction(email)}]

    try:
        response = await request_chat_completion(
            model=model,
            messages=messages,
            system=_SYSTEM_PROMPT,
            api_key=api_key,
            tools=[_EXTRACT_TOOL_SCHEMA],
        )
    except Exception as exc:
        logger.error(
            "Knowledge graph extraction API call failed",
            extra={"email_id": email.id, "error": str(exc)},
        )
        return None

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []

    for tool_call in tool_calls:
        fn = tool_call.get("function") or {}
        if fn.get("name") != _EXTRACT_TOOL_NAME:
            continue
        args = _coerce_arguments(fn.get("arguments"))
        if args is None:
            # _coerce_arguments already logged the raw preview
            logger.warning(
                "KG extraction failed; email will not be retried",
                extra={"email_id": email.id},
            )
            return None
        return _parse_extraction(args)

    logger.debug("KG extraction produced no tool call", extra={"email_id": email.id})
    return None


def _coerce_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    import ast

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
        # Fallback: some models return Python-style dicts (single quotes, trailing commas)
        try:
            result = ast.literal_eval(stripped)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
        logger.warning(
            "KG extraction: unparseable tool arguments (first 300 chars)",
            extra={"raw_preview": stripped[:300]},
        )
        return None
    return None


def _parse_extraction(args: Dict[str, Any]) -> ExtractionResult:
    entities: List[ExtractedEntity] = []

    for raw in args.get("entities") or []:
        if not isinstance(raw, dict):
            continue
        node_type = (raw.get("node_type") or "").strip()
        canonical_name = (raw.get("canonical_name") or "").strip()
        if not node_type or not canonical_name:
            continue

        facts: List[ExtractedFact] = []
        for rf in raw.get("facts") or []:
            if not isinstance(rf, dict):
                continue
            key = (rf.get("key") or "").strip()
            value = (rf.get("value") or "").strip()
            if not key or not value:
                continue
            try:
                confidence = float(rf.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            confidence = max(0.0, min(1.0, confidence))
            facts.append(ExtractedFact(key=key, value=value, confidence=confidence))

        entities.append(ExtractedEntity(
            node_type=node_type,
            canonical_name=canonical_name,
            facts=facts,
        ))

    relationships: List[ExtractedRelationship] = []
    for raw in args.get("relationships") or []:
        if not isinstance(raw, dict):
            continue
        from_entity = (raw.get("from_entity") or "").strip()
        to_entity = (raw.get("to_entity") or "").strip()
        edge_type = (raw.get("edge_type") or "").strip()
        if not from_entity or not to_entity or not edge_type:
            continue
        try:
            confidence = float(raw.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))
        raw_props = raw.get("edge_properties")
        edge_properties = (
            {str(k): str(v) for k, v in raw_props.items()}
            if isinstance(raw_props, dict) and raw_props
            else None
        )
        relationships.append(ExtractedRelationship(
            from_entity=from_entity,
            to_entity=to_entity,
            edge_type=edge_type,
            confidence=confidence,
            edge_properties=edge_properties,
        ))

    return ExtractionResult(entities=entities, relationships=relationships)


__all__ = [
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedFact",
    "ExtractedRelationship",
    "extract_from_email",
]
