"""Thematic momentum tracking.

Measures velocity and acceleration of topic coverage across the newsletter
corpus using rolling 7-day windows stored in kg_topic_attention.

Acceleration = (recent daily rate) / (prior daily rate)
where recent = last 7 days, prior = 8–14 days ago.

A topic is surfaced as a momentum signal only when:
    1. recent_mentions ≥ MIN_MENTIONS_FOR_ALERT   (absolute volume floor)
    2. recent_sources  ≥ MIN_SOURCES_FOR_ALERT    (multi-source credibility)
    3. acceleration    ≥ MOMENTUM_ACCELERATION_THRESHOLD  (velocity increase)

All three guards must pass. This is deliberately conservative — the user
wants fewer, more valuable signals, not more alerts.

Surfacing goes through InteractionAgentRuntime.handle_agent_message(), which
applies the Interaction Agent's own noise filter before reaching the user.
This path is identical to ImportantEmailWatcher._dispatch_summary() and must
not be bypassed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from ..store import KnowledgeGraphStore
from .store_ext import NewsletterGraphStore
from ....logging_config import logger

if TYPE_CHECKING:
    from ....agents.interaction_agent.runtime import InteractionAgentRuntime

# ---- Thresholds (conservative) -----------------------------------------------

# Topic coverage must have grown by this factor (recent vs prior 7-day window)
MOMENTUM_ACCELERATION_THRESHOLD = 3.0

# Minimum novel mentions in the recent 7-day window before considering alert
MIN_MENTIONS_FOR_ALERT = 5

# Minimum distinct publications in the recent window for credibility
MIN_SOURCES_FOR_ALERT = 2

# Maximum distinct topics surfaced in a single momentum check
_MAX_ALERTS_PER_CHECK = 3


@dataclass
class AcceleratingTopic:
    topic_node_id: int
    canonical_name: str
    recent_mentions: int
    prior_mentions: int
    acceleration: float
    recent_sources: int


def _resolve_interaction_runtime() -> "InteractionAgentRuntime":
    from ....agents.interaction_agent.runtime import InteractionAgentRuntime
    return InteractionAgentRuntime()


async def check_and_surface_momentum(
    kg_store: KnowledgeGraphStore,
    nl_store: NewsletterGraphStore,
) -> List[AcceleratingTopic]:
    """Identify accelerating topics and dispatch credible signals.

    Returns the list of accelerating topics (may be empty). Each qualifying
    topic is dispatched once per check via handle_agent_message(); the
    Interaction Agent decides whether to pass it to the user.
    """
    accelerating = _find_accelerating_topics(kg_store, nl_store)

    for topic in accelerating:
        try:
            await _dispatch_momentum_alert(topic)
        except Exception as exc:
            logger.exception(
                "Failed to dispatch momentum alert [topic=%s]: %s",
                topic.canonical_name,
                exc,
            )

    return accelerating


def _find_accelerating_topics(
    kg_store: KnowledgeGraphStore,
    nl_store: NewsletterGraphStore,
) -> List[AcceleratingTopic]:
    """Compute acceleration for all tracked topics and return qualifying ones."""
    candidate_ids = nl_store.get_all_tracked_topic_ids(
        min_recent_mentions=MIN_MENTIONS_FOR_ALERT,
    )
    accelerating: List[AcceleratingTopic] = []

    for topic_id in candidate_ids:
        stats = nl_store.get_topic_window_stats(topic_id, recent_days=7, prior_days=7)

        recent = stats["recent_mentions"]
        prior = stats["prior_mentions"]
        sources = stats["recent_sources"]

        if recent < MIN_MENTIONS_FOR_ALERT:
            continue
        if sources < MIN_SOURCES_FOR_ALERT:
            continue

        # Add 0.5 to prior to prevent zero-division and to require a meaningful
        # baseline before claiming 3× acceleration. A topic going 0 → 5 is
        # genuinely new, but could also be a single unusually busy week; the
        # absolute floor (MIN_MENTIONS) + multi-source guard (MIN_SOURCES) are
        # the primary quality controls for the zero-prior case.
        acceleration = recent / max(prior, 0.5)

        if acceleration < MOMENTUM_ACCELERATION_THRESHOLD:
            continue

        name = nl_store.get_node_canonical_name(topic_id)
        if name is None:
            continue

        accelerating.append(AcceleratingTopic(
            topic_node_id=topic_id,
            canonical_name=name,
            recent_mentions=recent,
            prior_mentions=prior,
            acceleration=round(acceleration, 1),
            recent_sources=sources,
        ))

    # Sort by acceleration descending, cap at _MAX_ALERTS_PER_CHECK
    accelerating.sort(key=lambda t: t.acceleration, reverse=True)
    return accelerating[:_MAX_ALERTS_PER_CHECK]


async def _dispatch_momentum_alert(topic: AcceleratingTopic) -> None:
    """Dispatch a momentum alert through the Interaction Agent noise filter.

    Message prefix "Newsletter intelligence:" distinguishes these from
    ImportantEmailWatcher notifications in conversation history.
    """
    runtime = _resolve_interaction_runtime()

    prior_str = str(topic.prior_mentions) if topic.prior_mentions > 0 else "none"
    message = (
        f"Newsletter intelligence: [MOMENTUM SIGNAL] "
        f"Topic '{topic.canonical_name}' has accelerated {topic.acceleration}× in newsletter "
        f"coverage this week — {topic.recent_mentions} novel mentions across "
        f"{topic.recent_sources} publications (vs {prior_str} the prior week). "
        f"This may be an early signal worth tracking."
    )

    await runtime.handle_agent_message(message)
    logger.info(
        "Momentum alert dispatched",
        extra={
            "topic": topic.canonical_name,
            "acceleration": topic.acceleration,
            "recent_mentions": topic.recent_mentions,
            "sources": topic.recent_sources,
        },
    )


__all__ = [
    "AcceleratingTopic",
    "MOMENTUM_ACCELERATION_THRESHOLD",
    "MIN_MENTIONS_FOR_ALERT",
    "MIN_SOURCES_FOR_ALERT",
    "check_and_surface_momentum",
]
