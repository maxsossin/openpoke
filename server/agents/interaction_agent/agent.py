"""Interaction agent helpers for prompt construction."""

from pathlib import Path
from typing import Dict, List
from ...services.execution.hot_cache import get_hot_cache, touch_agent
from html import escape

from ...services.execution import get_agent_roster

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(f"<active_agents>\n{_render_active_agents()}\n</active_agents>")
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# Format currently active execution agents into XML tags for LLM awareness
def _render_active_agents() -> str:
    """Render hot-cached agents only. Full roster available via roster_search tool."""
    hot = get_hot_cache()
    if not hot:
        return "None — use roster_search to find existing agents."

    lines = []
    for a in hot:
        name = escape(a["agent_name"], quote=True)
        status = escape(a.get("status", "unknown"), quote=True)
        summary = escape(a.get("summary", ""), quote=True)
        snippet = a.get("last_output_snippet", "")
        snippet_attr = f' last_output="{escape(snippet[:120], quote=True)}"' if snippet else ""
        lines.append(f'<agent name="{name}" status="{status}" summary="{summary}"{snippet_attr} />')

    return "\n".join(lines)


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
