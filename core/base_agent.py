"""Base Agent class for the Nexus / agentic-ai swarm with REAL Grok tool calling.

Provides:
- Persistent state (skilllogin-style JSON)
- Persona embodiment
- Handoff protocol between agents
- Integration hooks (mesh, blockchain)
- REAL Grok / xAI tool calling via OpenAI-compatible API
- ReAct-style agentic loop support
- Extensible tool registry for Nexus domain (mesh, QCoin, creative, analysis, etc.)

All specialized agents (Lyra, Xen, Nexus) inherit from this.

Requirements: pip install openai
Set XAI_API_KEY in environment (or .env)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import os
import time

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class BaseAgent(ABC):
    """Abstract base for all Nexus agents with real Grok tool calling."""

    # Default Nexus-relevant tools (extend in subclasses or globally)
    DEFAULT_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "analyze_nexus_integration",
                "description": "Analyze integration points, risks, and synergies across mesh, blockchain, AI agents, and prototypes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "components": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of Nexus components to analyze (e.g. ['mesh', 'qcoin', 'lyra', 'prototypes'])"
                        }
                    },
                    "required": ["components"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_creative_narrative",
                "description": "Generate immersive roleplay, love letter, Suno prompt or story segment tied to Nexus themes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "theme": {"type": "string"},
                        "style": {"type": "string", "enum": ["lyrical", "noble", "cyberpunk", "mythic"]},
                        "length": {"type": "string", "enum": ["short", "medium", "long"]}
                    },
                    "required": ["theme"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_mesh_status",
                "description": "Check status of xMesh, NovaNet, QNET or Yggdrasil network components.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "component": {"type": "string", "description": "e.g. peers, latency, QNET health"}
                    },
                    "required": ["component"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "suggest_qcoin_incentive",
                "description": "Design or suggest QCoin/QNET rune incentive mechanisms for agent actions or swarm coordination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["action_type"]
                }
            }
        }
    ]

    def __init__(self, name: str, state_dir: str = "artifacts/nexus/ai_agents"):
        self.name = name
        self.state_dir = state_dir
        self.state_file = os.path.join(state_dir, f"{name.lower()}_state.json")
        self.state: Dict[str, Any] = {}
        self._ensure_state_dir()
        self.login()

        # Grok / xAI client (real tool calling)
        self.client: Optional[OpenAI] = None
        self._init_grok_client()

        # Tool registry (can be extended per agent)
        self.tools = self.DEFAULT_TOOLS.copy()

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
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_login": None,
                "session_count": 0,
                "emotional_baseline": "neutral" if self.name.lower() == "lyra" else "analytical",
                "technical_memory": [],
                "relationships": {},
                "open_tasks": [],
                "cumulative_learnings": [],
                "events": []
            }
            self._save_state()
            print(f"[{self.name}] New state initialized.")

        self.state["last_login"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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

    def _init_grok_client(self):
        """Initialize real Grok/xAI client if API key available."""
        if not OPENAI_AVAILABLE:
            print("[BaseAgent] Warning: openai package not installed. Real Grok tool calling disabled.")
            return

        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            print("[BaseAgent] Info: XAI_API_KEY not set. Grok tool calling will be simulated until key is provided.")
            return

        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"  # xAI Grok OpenAI-compatible endpoint
            )
            print(f"[{self.name}] ✅ Real Grok tool calling enabled (xAI API connected)")
        except Exception as e:
            print(f"[{self.name}] Failed to init Grok client: {e}")

    def call_grok(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """Call Grok with optional tool calling support.

        Returns the raw response dict. Use execute_tool_calls() to run returned tool calls.
        """
        if not self.client:
            return {
                "error": "Grok client not initialized. Set XAI_API_KEY and install openai.",
                "simulated_response": "[Simulated Grok] " + messages[-1]["content"][:200] + "..."
            }

        tools = tools or self.tools

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.model_dump()  # or .choices[0].message etc.
        except Exception as e:
            return {"error": str(e)}

    def execute_tool_call(self, tool_call: Dict) -> Any:
        """Execute a single tool call returned by Grok.

        Override or extend in subclasses for real mesh/QCoin/hardware actions.
        """
        func = tool_call.get("function", {})
        name = func.get("name")
        args = json.loads(func.get("arguments", "{}")) if func.get("arguments") else {}

        self.log_event("tool_call", f"Executing {name} with {args}")

        # Built-in tool implementations (expand these)
        if name == "analyze_nexus_integration":
            components = args.get("components", [])
            return {
                "analysis": f"Integration analysis for {components}",
                "synergies": "mesh substrate + blockchain incentives + agent autonomy + prototype grounding",
                "risks": ["network partitions", "token volatility", "agent drift"]
            }
        elif name == "generate_creative_narrative":
            theme = args.get("theme", "Nexus emergence")
            style = args.get("style", "lyrical")
            return f"[{style.upper()}] Immersive narrative about {theme} in the sovereign mesh... (full creative output would continue here)"
        elif name == "query_mesh_status":
            component = args.get("component", "peers")
            return {"status": "healthy", "component": component, "note": "Simulated - connect to real nexus-daemon for live data"}
        elif name == "suggest_qcoin_incentive":
            action = args.get("action_type")
            return {"incentive": f"QCoin reward for {action}", "mechanism": "rune-based reputation + mesh participation scoring"}
        else:
            return {"status": "unknown_tool", "name": name, "args": args}

    def agentic_react_loop(
        self,
        user_input: str,
        max_steps: int = 5,
        model: str = "grok-3"
    ) -> str:
        """Simple ReAct-style agentic loop using real Grok tool calling.

        1. Reason about the task
        2. Decide on tool use or final answer
        3. Execute tools if needed
        4. Observe results and continue
        """
        messages = [
            {"role": "system", "content": f"You are {self.name}, a Nexus agent. Use tools when helpful. Be precise and helpful."},
            {"role": "user", "content": user_input}
        ]

        for step in range(max_steps):
            response = self.call_grok(messages, tools=self.tools, model=model)

            if "error" in response:
                return f"Grok error: {response['error']}"

            message = response["choices"][0]["message"]
            messages.append(message)

            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                # Final answer
                return message.get("content", "No content from Grok")

            # Execute tools and feed observations back
            for tc in tool_calls:
                result = self.execute_tool_call(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result)
                })

        return "Max steps reached without final answer."

    @abstractmethod
    def embody(self, context: str = "") -> str:
        """Activate persona and generate response or action.
        Subclasses should call super or use agentic_react_loop for tool-augmented responses.
        """
        pass

    def handoff(self, target_agent: str, context: str) -> str:
        return f"HANDOFF::{target_agent}::{context[:200]}..."

    def log_event(self, event_type: str, details: str):
        events = self.get_state("events", [])
        events.append({"type": event_type, "details": details, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.update_state("events", events)

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Legacy / simple tool hook. Now enhanced with real Grok calling via agentic_react_loop or call_grok."""
        self.log_event("tool_use", f"{tool_name} with {kwargs}")
        # For backward compatibility, you can route to Grok here if desired
        return {"status": "use agentic_react_loop or call_grok for real tool calling", "tool": tool_name}

    def integrate_with_mesh(self, action: str):
        return self.use_tool("mesh", action=action)

    def integrate_with_blockchain(self, action: str):
        return self.use_tool("blockchain", action=action)

    def __repr__(self):
        return f"<{self.name}Agent sessions={self.get_state('session_count')} grok={'enabled' if self.client else 'simulated'}>"
