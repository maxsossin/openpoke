"""Newsletter intelligence subsystem.

Signal/noise separation, cross-publication narrative threading,
thematic momentum tracking, and contrarian detection — implemented as
a unified pipeline that extends the existing knowledge graph.
"""

from .classifier import NewsletterClassification, classify_newsletter
from .extractor import NewsletterIntelligence, extract_newsletter_intelligence
from .novelty import NOVELTY_THRESHOLD, compute_fact_novelty_score, compute_topic_novelty
from .story_threader import STORY_MIN_SOURCES, StoryThreadResult, thread_story
from .momentum import (
    AcceleratingTopic,
    MOMENTUM_ACCELERATION_THRESHOLD,
    MIN_MENTIONS_FOR_ALERT,
    check_and_surface_momentum,
)
from .contrarian import (
    ContrarianAssessment,
    CONTRARIAN_MIN_PREVAILING_SOURCES,
    CONTRARIAN_MIN_DISSENTER_EVIDENCE,
    detect_contrarian_position,
)
from .processor import (
    NewsletterIntelligenceProcessor,
    NewsletterProcessingResult,
    get_newsletter_processor,
)
from .store_ext import NewsletterGraphStore, get_newsletter_graph_store

__all__ = [
    "NewsletterClassification",
    "classify_newsletter",
    "NewsletterIntelligence",
    "extract_newsletter_intelligence",
    "NOVELTY_THRESHOLD",
    "compute_fact_novelty_score",
    "compute_topic_novelty",
    "STORY_MIN_SOURCES",
    "StoryThreadResult",
    "thread_story",
    "AcceleratingTopic",
    "MOMENTUM_ACCELERATION_THRESHOLD",
    "MIN_MENTIONS_FOR_ALERT",
    "check_and_surface_momentum",
    "ContrarianAssessment",
    "CONTRARIAN_MIN_PREVAILING_SOURCES",
    "CONTRARIAN_MIN_DISSENTER_EVIDENCE",
    "detect_contrarian_position",
    "NewsletterIntelligenceProcessor",
    "NewsletterProcessingResult",
    "get_newsletter_processor",
    "NewsletterGraphStore",
    "get_newsletter_graph_store",
]
