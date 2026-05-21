"""Knowledge graph service: extraction, storage, and query."""

from .store import CONFIDENCE_THRESHOLD, KnowledgeGraphStore, get_knowledge_graph_store
from .watcher import KnowledgeGraphWatcher, get_knowledge_graph_watcher
from .newsletter.store_ext import NewsletterGraphStore, get_newsletter_graph_store
from .newsletter.processor import (
    NewsletterIntelligenceProcessor,
    NewsletterProcessingResult,
    get_newsletter_processor,
)

__all__ = [
    "CONFIDENCE_THRESHOLD",
    "KnowledgeGraphStore",
    "get_knowledge_graph_store",
    "KnowledgeGraphWatcher",
    "get_knowledge_graph_watcher",
    "NewsletterGraphStore",
    "get_newsletter_graph_store",
    "NewsletterIntelligenceProcessor",
    "NewsletterProcessingResult",
    "get_newsletter_processor",
]
