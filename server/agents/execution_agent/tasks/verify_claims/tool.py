"""Claim verification tools for the claim-verifier execution agent.

These tools are synchronous — all KG operations use threading.Lock, no async I/O.

The agent uses them in three phases per run:
  1. fetch_pending_claims  — select the batch of claims to evaluate
  2. query_claim_evidence  — gather evidence per claim (parallelisable)
  3. record_claim_verdict  — commit verdicts and update credibility (parallelisable)

Evidence ceiling per claim (enforced here, not in the LLM prompt):
  - Story framings:  up to _MAX_FRAMINGS, truncated to _FRAMING_TRUNCATE_CHARS
  - KG entity facts: up to _MAX_KG_ENTITIES, 5 facts each
  - Related claims:  up to _MAX_RELATED_CLAIMS
This prevents context overflow when a story has dozens of framings.

Verdict atomicity:
  - correct / incorrect → mark_claim_verified(): status update + credibility update
    in one lock acquisition. Atomicity guaranteed by the threading.Lock shared
    across NewsletterGraphStore and KnowledgeGraphStore.
  - expired            → expire_claim(): status update only, no credibility change.
  - pending            → mark_claim_last_checked(): stamps last_checked_at only.

The threading.Lock prevents concurrent modification from other threads in this
process. SQLite WAL mode prevents dirty reads from external tools.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from server.logging_config import logger
from server.services.knowledge_graph.store import get_knowledge_graph_store
from server.services.knowledge_graph.newsletter.store_ext import get_newsletter_graph_store
from .schemas import FETCH_TOOL_NAME, EVIDENCE_TOOL_NAME, VERDICT_TOOL_NAME

# ---- Evidence ceilings --------------------------------------------------------

_DEFAULT_CLAIM_BATCH = 10
_MAX_CLAIM_BATCH = 15
_MAX_FRAMINGS = 10          # story framings per claim
_MAX_KG_ENTITIES = 5        # semantically matched KG nodes per claim
_MAX_RELATED_CLAIMS = 5     # other claims on the same story
_FRAMING_TRUNCATE_CHARS = 400

# Terminal statuses — these cannot be overwritten by the verifier
_TERMINAL_STATUSES = frozenset({"correct", "incorrect", "expired"})

# Confidence floor below which correct/incorrect is downgraded to pending
_MIN_TERMINAL_CONFIDENCE = 0.70


# ---- Tool implementations ----------------------------------------------------


def fetch_pending_claims(limit: int = _DEFAULT_CLAIM_BATCH) -> Dict[str, Any]:
    """Return pending claims that have not been evaluated recently.

    The response includes today's date in ISO format so the agent can reason
    about claim time horizons without an additional tool call.
    """
    limit = max(1, min(int(limit), _MAX_CLAIM_BATCH))
    nl_store = get_newsletter_graph_store()

    try:
        claims = nl_store.get_pending_claims(limit=limit)
    except Exception as exc:
        logger.exception("fetch_pending_claims: DB query failed: %s", exc)
        return {"error": str(exc), "claims": [], "count": 0}

    today = datetime.now(timezone.utc).date().isoformat()
    today_date = datetime.now(timezone.utc).date()

    enriched: List[Dict[str, Any]] = []
    for c in claims:
        days_since: Optional[int] = None
        if c.get("claim_date"):
            try:
                claim_date = datetime.fromisoformat(c["claim_date"]).date()
                days_since = (today_date - claim_date).days
            except (ValueError, TypeError):
                pass

        enriched.append({
            "id": c["id"],
            "claim_text": c["claim_text"],
            "claim_date": c.get("claim_date"),
            "story_node_id": c.get("story_node_id"),
            "story_name": c.get("story_name") or "(no story)",
            "source_node_id": c.get("source_node_id"),
            "source_name": c.get("source_name") or "(unknown source)",
            "days_since_claim": days_since,
            "last_checked_at": c.get("last_checked_at"),
        })

    logger.debug(
        "fetch_pending_claims returned %d claims",
        len(enriched),
        extra={"limit": limit},
    )
    return {
        "today": today,
        "claims": enriched,
        "count": len(enriched),
    }


def query_claim_evidence(
    claim_id: int,
    story_node_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Gather evidence for a single claim from the knowledge graph.

    Evidence is drawn from three sources:
      1. Story framings — perspectives on the story from other publications.
         These are the primary signal for correct/incorrect verdicts.
      2. Semantically related KG entities — facts extracted from emails that
         are semantically close to the claim text. Background context only.
      3. Other claims on the same story — supplementary context; do not
         substitute for primary framing evidence.

    All lists are bounded to prevent context overflow. Framing text is
    truncated to _FRAMING_TRUNCATE_CHARS characters.
    """
    kg_store = get_knowledge_graph_store()
    nl_store = get_newsletter_graph_store()

    # Fetch the claim record (works regardless of verification status)
    try:
        claim_row = nl_store.get_claim_by_id(int(claim_id))
    except Exception as exc:
        logger.exception("query_claim_evidence: claim fetch failed [id=%s]: %s", claim_id, exc)
        return {"error": f"Failed to fetch claim {claim_id}: {exc}"}

    if claim_row is None:
        return {"error": f"Claim {claim_id} not found"}

    effective_story_node_id: Optional[int] = (
        story_node_id
        or claim_row.get("story_node_id")
    )
    claim_text: str = claim_row.get("claim_text", "")

    evidence: Dict[str, Any] = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_date": claim_row.get("claim_date"),
        "source_name": claim_row.get("source_name") or "(unknown)",
        "story_name": claim_row.get("story_name") or "(no story)",
        "story_node_id": effective_story_node_id,
        "current_status": claim_row.get("verification_status"),
        "story_framings": [],
        "related_claims": [],
        "kg_entities": [],
    }

    # --- Evidence 1: story framings (primary signal) --------------------------
    if effective_story_node_id is not None:
        try:
            framings = nl_store.get_story_framings(
                effective_story_node_id,
                limit=_MAX_FRAMINGS,
                since_days=365,
            )
            evidence["story_framings"] = [
                {
                    "source": f.get("source_name", "unknown"),
                    "framing": (f.get("framing_text") or "")[:_FRAMING_TRUNCATE_CHARS],
                    "sentiment": f.get("sentiment", "neutral"),
                    "date": f.get("source_email_ts", ""),
                    "key_claim": (f.get("key_claim") or "")[:200],
                    "claim_accuracy": f.get("claim_accuracy"),
                }
                for f in framings
            ]
        except Exception as exc:
            logger.warning(
                "query_claim_evidence: story framings failed [claim=%d]: %s", claim_id, exc
            )

    # --- Evidence 2: other claims on the same story ---------------------------
    if effective_story_node_id is not None:
        try:
            related = nl_store.get_related_claims_for_story(
                effective_story_node_id,
                exclude_claim_id=int(claim_id),
                limit=_MAX_RELATED_CLAIMS,
            )
            evidence["related_claims"] = [
                {
                    "id": r["id"],
                    "claim_text": r["claim_text"],
                    "claim_date": r.get("claim_date"),
                    "status": r.get("verification_status"),
                    "source": r.get("source_name") or "unknown",
                }
                for r in related
            ]
        except Exception as exc:
            logger.warning(
                "query_claim_evidence: related claims failed [claim=%d]: %s", claim_id, exc
            )

    # --- Evidence 3: semantically related KG entities (background context) ----
    if claim_text:
        try:
            candidates = kg_store.query_entities_semantically(
                claim_text, n_results=_MAX_KG_ENTITIES
            )
            kg_hits: List[Dict[str, Any]] = []
            for candidate in candidates[:_MAX_KG_ENTITIES]:
                node = kg_store.query_node_by_name(
                    candidate["canonical_name"],
                    node_type=candidate.get("node_type") or None,
                )
                if node:
                    kg_hits.append({
                        "name": node["canonical_name"],
                        "type": node["node_type"],
                        "facts": [
                            {"key": f["fact_key"], "value": f["fact_value"]}
                            for f in node["facts"][:5]
                        ],
                        "similarity": candidate.get("similarity"),
                    })
            evidence["kg_entities"] = kg_hits
        except Exception as exc:
            logger.warning(
                "query_claim_evidence: KG entity query failed [claim=%d]: %s", claim_id, exc
            )

    framing_count = len(evidence["story_framings"])
    entity_count = len(evidence["kg_entities"])
    logger.debug(
        "query_claim_evidence assembled",
        extra={
            "claim_id": claim_id,
            "framings": framing_count,
            "kg_entities": entity_count,
            "related_claims": len(evidence["related_claims"]),
        },
    )
    return evidence


def record_claim_verdict(
    claim_id: int,
    verdict: str,
    confidence: float,
    reasoning: str,
) -> Dict[str, Any]:
    """Commit a verdict for a claim and update source credibility when appropriate.

    Verdict routing:
      correct   → mark_claim_verified(is_correct=True)
                  Atomically: verification_status + credibility counts in one lock.
      incorrect → mark_claim_verified(is_correct=False)
                  Same atomic path.
      expired   → expire_claim()
                  Status update only; credibility counts are NOT incremented.
                  Expiry is not an accuracy signal.
      pending   → mark_claim_last_checked()
                  Stamps last_checked_at; status unchanged; no credibility change.

    Safety guards enforced in code (not delegated to the LLM):
      - verdict not in allowed set → error returned, no write
      - confidence < _MIN_TERMINAL_CONFIDENCE and verdict in {correct, incorrect}
        → downgraded to pending automatically
      - current_status already terminal → skipped, not an error
    """
    nl_store = get_newsletter_graph_store()

    # --- Input validation -----------------------------------------------------
    verdict = str(verdict).lower().strip()
    if verdict not in {"correct", "incorrect", "expired", "pending"}:
        return {
            "error": (
                f"Invalid verdict '{verdict}'. "
                "Must be one of: correct, incorrect, expired, pending."
            )
        }

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return {"error": f"confidence must be a number, got: {confidence!r}"}

    if not (0.0 <= confidence <= 1.0):
        return {"error": f"confidence must be in [0.0, 1.0], got {confidence}"}

    # Enforce confidence floor for terminal verdicts — downgrade rather than reject
    # so the claim gets marked as last_checked and doesn't re-enter immediately.
    original_verdict = verdict
    if verdict in {"correct", "incorrect"} and confidence < _MIN_TERMINAL_CONFIDENCE:
        logger.info(
            "record_claim_verdict: confidence %.2f below floor %.2f; downgrading to pending",
            confidence,
            _MIN_TERMINAL_CONFIDENCE,
            extra={"claim_id": claim_id, "original_verdict": verdict},
        )
        verdict = "pending"

    # --- Guard: do not overwrite terminal statuses ----------------------------
    try:
        claim_row = nl_store.get_claim_by_id(int(claim_id))
    except Exception as exc:
        logger.exception(
            "record_claim_verdict: status check failed [claim=%d]: %s", claim_id, exc
        )
        return {"error": f"Failed to read claim status: {exc}"}

    if claim_row is None:
        return {"error": f"Claim {claim_id} not found"}

    current_status = claim_row.get("verification_status", "")
    if current_status in _TERMINAL_STATUSES:
        logger.debug(
            "record_claim_verdict: claim %d already terminal (%s); skipping",
            claim_id,
            current_status,
        )
        return {
            "status": "skipped",
            "claim_id": claim_id,
            "reason": f"Already has terminal status '{current_status}'",
        }

    # --- Write verdict --------------------------------------------------------
    novelty_score = claim_row.get("novelty_score")
    novelty_score_float = float(novelty_score) if novelty_score is not None else None

    credibility_updated = False
    try:
        if verdict == "correct":
            nl_store.mark_claim_verified(
                claim_id, is_correct=True, novelty_score=novelty_score_float
            )
            credibility_updated = True
        elif verdict == "incorrect":
            nl_store.mark_claim_verified(
                claim_id, is_correct=False, novelty_score=novelty_score_float
            )
            credibility_updated = True
        elif verdict == "expired":
            nl_store.expire_claim(claim_id)
        else:  # pending
            nl_store.mark_claim_last_checked(claim_id)
    except Exception as exc:
        logger.exception(
            "record_claim_verdict: DB write failed [claim=%d, verdict=%s]: %s",
            claim_id,
            verdict,
            exc,
        )
        return {"error": f"Failed to record verdict: {exc}"}

    logger.info(
        "Claim verdict recorded",
        extra={
            "claim_id": claim_id,
            "verdict": verdict,
            "original_verdict": original_verdict,
            "confidence": round(confidence, 3),
            "credibility_updated": credibility_updated,
            "source_name": claim_row.get("source_name"),
        },
    )

    result: Dict[str, Any] = {
        "status": "recorded",
        "claim_id": claim_id,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "credibility_updated": credibility_updated,
        "reasoning": reasoning,
    }
    if original_verdict != verdict:
        result["note"] = (
            f"Original verdict '{original_verdict}' downgraded to 'pending' "
            f"because confidence {confidence:.2f} < {_MIN_TERMINAL_CONFIDENCE:.2f}."
        )

    # Emit a source credibility snapshot after a terminal verdict so the
    # execution agent's final report can include the updated accuracy rate.
    if credibility_updated and claim_row.get("source_node_id"):
        try:
            nl_store_ref = nl_store
            cred = nl_store_ref.get_source_credibility(int(claim_row["source_node_id"]))
            result["source_credibility_after"] = cred
        except Exception:
            pass  # Non-critical; don't fail the verdict write

    return result


# ---- Registry ----------------------------------------------------------------


def build_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:  # noqa: ARG001
    """Return verifier tool callables. agent_name unused; present for registry protocol."""
    return {
        FETCH_TOOL_NAME: fetch_pending_claims,
        EVIDENCE_TOOL_NAME: query_claim_evidence,
        VERDICT_TOOL_NAME: record_claim_verdict,
    }


__all__ = ["build_registry", "get_schemas"]
