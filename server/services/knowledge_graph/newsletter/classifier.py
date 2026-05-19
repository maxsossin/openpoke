"""Newsletter detection and publication source fingerprinting.

Heuristic-first classification: no LLM call. Conservative by design —
prefer false negatives (newsletter treated as regular email) over false
positives (regular email treated as newsletter). A misclassified newsletter
falls back to the standard KG extraction path and loses newsletter
intelligence; a misclassified regular email would inject noise into the
newsletter subsystem.

Signal hierarchy:
    STRONG (one sufficient): Gmail category label, known publication domain
    WEAK   (two required):   automated sender local-part, newsletter keywords
                             in subject, unsubscribe text in body
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from ...gmail.processing import ProcessedEmail

# ---- Signal tables ----------------------------------------------------------

# Gmail labels that reliably indicate bulk/newsletter mail
_NEWSLETTER_LABELS: Set[str] = {
    "CATEGORY_PROMOTIONS",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
}

# Known publication domains → canonical publication name.
# Subdomain fallback: "mail.substack.com" matches "substack.com" entry.
_KNOWN_DOMAINS: Dict[str, str] = {
    "bloomberg.com":          "Bloomberg",
    "bloomberg.net":          "Bloomberg",
    "ft.com":                 "Financial Times",
    "nytimes.com":            "New York Times",
    "wsj.com":                "Wall Street Journal",
    "economist.com":          "The Economist",
    "theatlantic.com":        "The Atlantic",
    "newyorker.com":          "The New Yorker",
    "substack.com":           "Substack",
    "substackcdn.com":        "Substack",
    "mail.substack.com":      "Substack",
    "beehiiv.com":            "Beehiiv",
    "morningbrew.com":        "Morning Brew",
    "axios.com":              "Axios",
    "politico.com":           "Politico",
    "wired.com":              "Wired",
    "techcrunch.com":         "TechCrunch",
    "theinformation.com":     "The Information",
    "stratechery.com":        "Stratechery",
    "bensbites.com":          "Ben's Bites",
    "tldr.tech":              "TLDR",
    "hackernewsletter.com":   "Hacker Newsletter",
    "campaignmonitor.com":    "Campaign Monitor",
    "mailchimp.com":          "Mailchimp",
    "sendgrid.net":           "SendGrid",
    "constantcontact.com":    "Constant Contact",
    "klaviyo.com":            "Klaviyo",
}

# Sender local-parts that indicate automated newsletter delivery
_AUTOMATED_LOCAL_PART = re.compile(
    r"^(newsletter|noreply|no-reply|updates|digest|daily|weekly|news|"
    r"briefing|alert|notifications?|hello|team|info)[@+]",
    re.IGNORECASE,
)

# Subject substrings associated with newsletter issues
_NEWSLETTER_SUBJECT_TOKENS: Set[str] = {
    "newsletter", "digest", "weekly", "daily", "briefing", "edition",
    "roundup", "recap", "morning", "evening", "update", "bulletin",
    "issue #", "vol.", "volume", "#",
}

# Body text indicating an unsubscribe mechanism (CAN-SPAM / GDPR requirement)
_UNSUBSCRIBE_PATTERN = re.compile(
    r"\bunsubscribe\b|\bopt.?out\b|\bmanage.*?subscription|\bpreferences\b",
    re.IGNORECASE,
)

_MIN_STRONG = 1  # one strong signal is enough
_MIN_WEAK = 2    # need two weak signals when no strong signal


@dataclass
class NewsletterClassification:
    is_newsletter: bool
    confidence: float
    publication_name: str
    sender_domain: str
    sender_address: str
    signals: List[str] = field(default_factory=list)


def _extract_sender_parts(sender: str) -> Tuple[str, str]:
    """Return (email_address_lower, domain) from a raw sender string."""
    angle = re.search(r"<([^>]+)>", sender)
    address = (angle.group(1) if angle else sender).strip().lower()
    domain_match = re.search(r"@([\w.-]+)\s*$", address)
    domain = domain_match.group(1) if domain_match else ""
    return address, domain


def _resolve_publication_name(domain: str) -> str:
    """Canonical publication name from domain, with subdomain fallback."""
    if domain in _KNOWN_DOMAINS:
        return _KNOWN_DOMAINS[domain]
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in _KNOWN_DOMAINS:
            return _KNOWN_DOMAINS[candidate]
    # Fall back: second-level domain, capitalized
    return parts[-2].capitalize() if len(parts) >= 2 else domain.capitalize()


def classify_newsletter(email: ProcessedEmail) -> NewsletterClassification:
    """Classify an email as newsletter or not using layered heuristics.

    No LLM call. Returns a NewsletterClassification whose is_newsletter
    field gates the entire newsletter intelligence pipeline.
    """
    address, domain = _extract_sender_parts(email.sender)
    publication_name = _resolve_publication_name(domain)
    signals: List[str] = []
    strong = 0
    weak = 0

    # Strong signal: Gmail category label
    label_hits = set(email.label_ids or []) & _NEWSLETTER_LABELS
    if label_hits:
        signals.append(f"label:{','.join(sorted(label_hits))}")
        strong += 1

    # Strong signal: known publication domain
    if domain in _KNOWN_DOMAINS or any(
        domain.endswith("." + d) for d in _KNOWN_DOMAINS
    ):
        signals.append(f"known_domain:{domain}")
        strong += 1

    # Weak signal: automated sender local-part
    local = address.split("@")[0] if "@" in address else address
    if _AUTOMATED_LOCAL_PART.match(local + "@"):
        signals.append(f"automated_sender:{local}")
        weak += 1

    # Weak signal: newsletter keywords in subject
    subject_lower = (email.subject or "").lower()
    matched_tokens = [t for t in _NEWSLETTER_SUBJECT_TOKENS if t in subject_lower]
    if matched_tokens:
        signals.append(f"subject_tokens:{','.join(matched_tokens[:3])}")
        weak += 1

    # Weak signal: unsubscribe text in body (check first 4 KB for speed)
    if _UNSUBSCRIBE_PATTERN.search((email.clean_text or "")[:4096]):
        signals.append("unsubscribe_in_body")
        weak += 1

    is_newsletter = strong >= _MIN_STRONG or weak >= _MIN_WEAK
    confidence = min(1.0, 0.5 * strong + 0.15 * weak) if is_newsletter else 0.0

    return NewsletterClassification(
        is_newsletter=is_newsletter,
        confidence=confidence,
        publication_name=publication_name,
        sender_domain=domain,
        sender_address=address,
        signals=signals,
    )


__all__ = ["NewsletterClassification", "classify_newsletter"]
