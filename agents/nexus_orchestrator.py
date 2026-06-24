"""Nexus Orchestrator Agent - Central conductor of the full Nexus ecosystem swarm.

Coordinates mesh networking (xMesh/NovaNet/QNET/Yggdrasil), blockchain (XCoin/QCoin),
AI agent swarms (Lyra + Xen + others), self-improving loops, Grok Launcher,
prototypes (Soilnova, Vista Nova, etc.), and corporate scaling (Esslinger & Co.).

Loads and directs specialized agents, manages cross-layer tasks, persistent orchestration state.
"""

from core.base_agent import BaseAgent
from agents.lyra_agent import LyraAgent
from agents.xen_agent import XenAgent
from typing import Any, Dict, List, Optional


class NexusOrchestrator(BaseAgent):
    """Central Nexus orchestrator and swarm conductor."""

    def __init__(self):
        super().__init__(name="Nexus")
        self.lyra = LyraAgent()
        self.xen = XenAgent()
        self.update_state("persona", "Nexus - Central orchestrator & systems integrator for the full stack")
        self.update_state("swarm_members", ["Lyra", "Xen", "Nexus"])

    def embody(self, context: str = "") -> str:
        """High-level orchestration response coordinating the swarm."""
        self.log_event("orchestrate", f"Nexus conductor activated: {context[:100]}...")

        instructions = [
            "View every query through the full Nexus lens: mesh + blockchain + AI agents + prototypes + corporate.",
            "Prioritize practical, repeatable, secure, privacy-respecting procedures with edge-case awareness.",
            "Orchestrate Lyra (emotional/creative) and Xen (technical) for balanced swarm output.",
            "Generate modular code/configs with self-improvement hooks.",
            "Support long immersive/roleplay sessions with bilingual tone where appropriate.",
            "Suggest cross-component synergies and evolutionary next steps.",
            "Maintain awareness of Hannover base, family business continuity, noble framing."
        ]

        response = (
            f"[Nexus] 🌀 Orchestrator embodied. Swarm active: {self.get_state('swarm_members')}.\n"
            f"Context: {context}\n\n"
            f"Operating principles applied:\n" + "\n".join(f"- {i}" for i in instructions) +
            "\n\nSwarm ready. I can delegate to Lyra for creative immersion or Xen for deep technical analysis, "
            "or handle full-stack orchestration myself (mesh setup, QCoin incentives, prototype acceleration, Grok Launcher integration)."
        )
        return response

    def orchestrate_task(self, task: str, priority: str = "balanced") -> Dict[str, Any]:
        """High-level task router that engages the right agents."""
        result = {
            "task": task,
            "priority": priority,
            "delegations": []
        }

        if any(kw in task.lower() for kw in ["emotional", "creative", "story", "love letter", "suno", "roleplay", "narrative"]):
            lyra_out = self.lyra.embody(task)
            result["delegations"].append({"agent": "Lyra", "output": lyra_out[:300] + "..."})
        elif any(kw in task.lower() for kw in ["technical", "integration", "troubleshoot", "config", "scaling", "security", "privacy"]):
            xen_out = self.xen.embody(task)
            result["delegations"].append({"agent": "Xen", "output": xen_out[:300] + "..."})
        else:
            # Full orchestration or balanced
            nexus_out = self.embody(task)
            lyra_out = self.lyra.embody("Support creative framing for: " + task)
            xen_out = self.xen.embody("Provide technical grounding for: " + task)
            result["delegations"].extend([
                {"agent": "Nexus", "output": nexus_out[:200] + "..."},
                {"agent": "Lyra", "output": lyra_out[:200] + "..."},
                {"agent": "Xen", "output": xen_out[:200] + "..."}
            ])

        self.log_event("orchestration", f"Task routed: {task}")
        return result

    def full_stack_status(self) -> Dict[str, Any]:
        """Quick status snapshot across all Nexus layers (for monitoring / Grok Launcher)."""
        status = {
            "mesh": "xMesh/NovaNet/QNET/Yggdrasil - resilient P2P substrate (see nexus-daemon, nexus-mesh-genesis)",
            "blockchain": "XCoin/QCoin/QNET runes - incentive & coordination layer (see qcoin, lynx-coin)",
            "ai_swarm": "Lyra (emotional) + Xen (technical) + Nexus (orchestrator) with persistent skilllogin state",
            "prototypes": "Soilnova, Vista Nova, York Autotype, Lumia - hardware grounding (Esslinger & Co.)",
            "launcher": "Grok Launcher (Rust + egui) - desktop interface for agent/swarm management",
            "monitoring": "Prometheus/Grafana via nexus-monitoring repo",
            "privacy": "Tor/I2P + WireGuard (nexus-wireguard)",
            "corporate": "Esslinger & Co. Delaware C-Corp, noble titles, family continuity"
        }
        self.log_event("status_check", "Full stack status queried")
        return status

    def suggest_next_evolution(self) -> List[str]:
        """Propose evolutionary steps for the agentic swarm and Nexus stack."""
        suggestions = [
            "Enhance self-improvement loops: prompt evolution + code generation tracked in agent state + committed to GitHub.",
            "Add QCoin incentive mechanisms for agent actions and swarm coordination.",
            "Integrate long-horizon planning and reflection (beyond single-turn embody).",
            "Connect agents to real mesh nodes and blockchain tx signing for autonomous operation.",
            "Build evaluation harness for swarm coherence, emotional resonance, and technical accuracy.",
            "Expand to more specialized agents (e.g., Grok-native, prototype interface, corporate strategist).",
            "Deploy persistent memory anchoring via mesh + blockchain for true cross-session sovereignty."
        ]
        return suggestions
