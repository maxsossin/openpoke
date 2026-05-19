"""Session-scoped hot cache of recently active agents."""

from collections import OrderedDict

from .descriptor import load_descriptors

_HOT_CACHE_SIZE = 7
_session_cache: OrderedDict[str, dict] = OrderedDict()


def touch_agent(agent_name: str) -> None:
    """Promote an agent to the front of the hot cache with fresh data."""
    descriptors = load_descriptors()
    if agent_name in descriptors:
        d = descriptors[agent_name]
        entry = {
            "agent_name": agent_name,
            "summary": d.summary,
            "status": d.status,
            "last_output_snippet": d.last_output_snippet,
        }
    else:
        # Agent exists on roster but has no descriptor yet (just spawned)
        entry = {
            "agent_name": agent_name,
            "summary": "Newly created agent, no output yet.",
            "status": "active",
            "last_output_snippet": "",
        }

    _session_cache[agent_name] = entry
    _session_cache.move_to_end(agent_name)

    if len(_session_cache) > _HOT_CACHE_SIZE:
        _session_cache.popitem(last=False)


def get_hot_cache() -> list[dict]:
    """Return cached agents, most recent first."""
    return list(reversed(_session_cache.values()))


def reset_session_cache() -> None:
    """Call at conversation session boundaries."""
    _session_cache.clear()