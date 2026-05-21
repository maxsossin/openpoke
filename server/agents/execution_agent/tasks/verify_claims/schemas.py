"""Tool schemas for the claim verification task tools."""

from __future__ import annotations

from typing import Any, Dict, List

FETCH_TOOL_NAME = "fetch_pending_claims"
EVIDENCE_TOOL_NAME = "query_claim_evidence"
VERDICT_TOOL_NAME = "record_claim_verdict"

_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": FETCH_TOOL_NAME,
            "description": (
                "Return pending newsletter claims that have not been checked recently. "
                "Call this once at the start of each verification run. "
                "The response includes today's date for use in expiry reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of claims to return. "
                            "Defaults to 10; capped at 15."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": EVIDENCE_TOOL_NAME,
            "description": (
                "Gather evidence for a single pending claim from the knowledge graph. "
                "Returns story framings from multiple publications, semantically related "
                "KG entity facts, and other claims on the same story. "
                "Call this for each claim returned by fetch_pending_claims — "
                "all calls can be made simultaneously in a single iteration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "integer",
                        "description": "ID of the claim to gather evidence for.",
                    },
                    "story_node_id": {
                        "type": "integer",
                        "description": (
                            "Story node ID from the claim record. "
                            "Pass this to scope framing queries to the correct story."
                        ),
                    },
                },
                "required": ["claim_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": VERDICT_TOOL_NAME,
            "description": (
                "Record your verdict for a claim and update source credibility scores. "
                "Must be called for every claim returned by fetch_pending_claims. "
                "Terminal verdicts (correct, incorrect, expired) are permanent — "
                "only use them when evidence meets the criteria defined in your system prompt. "
                "All verdict calls can be made simultaneously in a single iteration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "integer",
                        "description": "ID of the claim being evaluated.",
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["correct", "incorrect", "expired", "pending"],
                        "description": (
                            "correct: claim confirmed by ≥2 independent sources, confidence ≥0.75. "
                            "incorrect: claim refuted by ≥2 independent sources, confidence ≥0.75. "
                            "expired: time horizon clearly passed (>14 days) with no resolution evidence. "
                            "pending: insufficient evidence; claim will be re-evaluated later."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": (
                            "Your confidence in the verdict, 0.0–1.0. "
                            "Correct/incorrect verdicts with confidence below 0.70 are "
                            "automatically downgraded to pending by the system."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "1–2 sentences explaining your verdict, citing specific evidence. "
                            "This is stored permanently and used for auditability."
                        ),
                    },
                },
                "required": ["claim_id", "verdict", "confidence", "reasoning"],
                "additionalProperties": False,
            },
        },
    },
]


def get_schemas() -> List[Dict[str, Any]]:
    return _SCHEMAS


__all__ = [
    "FETCH_TOOL_NAME",
    "EVIDENCE_TOOL_NAME",
    "VERDICT_TOOL_NAME",
    "get_schemas",
]
