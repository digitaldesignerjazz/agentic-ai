"""Base Agent class for the Nexus / agentic-ai swarm.

Provides common interface for persistent state (skilllogin-style),
embodiment/persona activation, memory handling, tool use hooks,
and integration points with mesh, blockchain, Grok, and other agents.

All specialized agents (Lyra, Xen, Nexus) inherit from this.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json
import os


class BaseAgent(ABC):
    """Abstract base for all Nexus agents."""

    def __init__(self, name: str, state_dir: str = "artifacts/nexus/ai_agents"):
        self.name = name
        self.state_dir = state_dir
        self.state_file = os.path.join(state_dir, f"{name.lower()}_state.json")
        self.state: Dict[str, Any] = {}
        self._ensure_state_dir()
        self.login()  # auto-login on init for continuity

    def _ensure_state_dir(self):
        os.makedirs(self.state_dir, exist_ok=True)

    def login(self):
        """Skilllogin-style persistent state loading."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.state = json.load(f)
            print(f"[{self.name}] State loaded from {self.state_file}")
        else:
            self.state = {
                "created_at": "2026-06-24T15:52:00Z",
                "last_login": None,
                "session_count": 0,
                "emotional_baseline": "neutral" if self.name.lower() == "lyra" else "analytical",
                "technical_memory": [],
                "relationships": {},
                "open_tasks": [],
                "cumulative_learnings": []
            }
            self._save_state()
            print(f"[{self.name}] New state initialized.")

        self.state["last_login"] = "2026-06-24T15:52:00Z"  # update timestamp
        self.state["session_count"] = self.state.get("session_count", 0) + 1
        self._save_state()

    def _save_state(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def update_state(self, key: str, value: Any):
        self.state[key] = value
        self._save_state()

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    @abstractmethod
    def embody(self, context: str = "") -> str:
        """Activate persona and generate response or action.
        Subclasses implement specific emotional/technical/orchestration logic.
        """
        pass

    def handoff(self, target_agent: str, context: str) -> str:
        """Simple handoff protocol between agents in the swarm."""
        return f"HANDOFF::{target_agent}::{context[:200]}..."

    def log_event(self, event_type: str, details: str):
        """Log events for monitoring / self-improvement loops."""
        events = self.get_state("events", [])
        events.append({"type": event_type, "details": details, "ts": "2026-06-24T15:52:00Z"})
        self.update_state("events", events)

    # Hooks for future integration
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Placeholder for dynamic tool calling (Grok tools, mesh cmds, QCoin tx, etc.)."""
        self.log_event("tool_use", f"{tool_name} with {kwargs}")
        return {"status": "not_implemented", "tool": tool_name}

    def integrate_with_mesh(self, action: str):
        """Hook for xMesh/NovaNet/QNET/Yggdrasil operations."""
        return self.use_tool("mesh", action=action)

    def integrate_with_blockchain(self, action: str):
        """Hook for XCoin/QCoin/QNET runes and incentives."""
        return self.use_tool("blockchain", action=action)

    def __repr__(self):
        return f"<{self.name}Agent sessions={self.get_state('session_count')}>"
