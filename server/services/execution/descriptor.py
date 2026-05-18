"""Agent descriptor persistence - structured metadata for execution agents."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from ...logging_config import logger

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DESCRIPTOR_PATH = _DATA_DIR / "execution_agents" / "descriptors.json"


@dataclass
class AgentDescriptor:
    agent_name: str
    summary: str
    tags: list[str]
    last_active: str        
    status: str             
    last_output_snippet: str


def load_descriptors() -> dict[str, AgentDescriptor]:
    if not _DESCRIPTOR_PATH.exists():
        return {}
    try:
        with open(_DESCRIPTOR_PATH) as f:
            raw = json.load(f)
        return {k: AgentDescriptor(**v) for k, v in raw.items()}
    except Exception as exc:
        logger.warning(f"Failed to load descriptors: {exc}")
        return {}


def save_descriptor(descriptor: AgentDescriptor) -> None:
    try:
        _DESCRIPTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = load_descriptors()
        existing[descriptor.agent_name] = descriptor
        with open(_DESCRIPTOR_PATH, "w") as f:
            json.dump({k: asdict(v) for k, v in existing.items()}, f, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to save descriptor for {descriptor.agent_name}: {exc}")