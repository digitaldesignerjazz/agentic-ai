"""Base Agent class for the Nexus / agentic-ai swarm with REAL Grok tool calling + full ASYNC support.

Features:
- Persistent state (skilllogin-style)
- Persona embodiment + handoff protocol
- REAL Grok / xAI tool calling (sync + async)
- ReAct-style agentic loops (sync + async)
- Extensible Nexus tool registry (mesh, QCoin, creative, integration analysis)
- Graceful fallback when no API key

All specialized agents (Lyra, Xen, Nexus) inherit this.

pip install openai

export XAI_API_KEY=sk-...
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import os
import time
import asyncio

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = AsyncOpenAI = None


class BaseAgent(ABC):
    """Abstract base for all Nexus agents with sync + async Grok tool calling."""

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
                            "description": "List of Nexus components to analyze"
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
                "description": "Generate immersive roleplay, love letter, Suno prompt or story tied to Nexus themes.",
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
                "description": "Check status of xMesh, NovaNet, QNET or Yggdrasil components.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "component": {"type": "string"}
                    },
                    "required": ["component"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "suggest_qcoin_incentive",
                "description": "Design QCoin/QNET rune incentive mechanisms for agent/swarm actions.",
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

        # Sync + Async Grok clients
        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        self._init_grok_clients()

        self.tools = self.DEFAULT_TOOLS.copy()

    def _ensure_state_dir(self):
        os.makedirs(self.state_dir, exist_ok=True)

    def login(self):
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

    def _init_grok_clients(self):
        """Initialize both sync and async Grok/xAI clients."""
        if not OPENAI_AVAILABLE:
            print("[BaseAgent] Warning: openai not installed. Grok calls will be simulated.")
            return

        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            print("[BaseAgent] Info: XAI_API_KEY not set → using simulated mode.")
            return

        try:
            self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            self.async_client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            print(f"[{self.name}] ✅ Grok tool calling enabled (sync + async)")
        except Exception as e:
            print(f"[{self.name}] Failed to init Grok clients: {e}")

    # ==================== SYNC GROK METHODS ====================

    def call_grok(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """Synchronous Grok call with tool support."""
        if not self.client:
            return {
                "error": "Grok client not ready. Set XAI_API_KEY.",
                "simulated": True,
                "content": "[Simulated] " + messages[-1].get("content", "")[:300]
            }

        tools = tools or self.tools
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.model_dump()
        except Exception as e:
            return {"error": str(e)}

    def execute_tool_call(self, tool_call: Dict) -> Any:
        """Execute a tool call returned by Grok (sync version). Override in subclasses for real actions."""
        func = tool_call.get("function", {})
        name = func.get("name")
        args = json.loads(func.get("arguments", "{}")) if func.get("arguments") else {}

        self.log_event("tool_call", f"{name}({args})")

        if name == "analyze_nexus_integration":
            return {"analysis": args.get("components"), "synergies": "mesh + blockchain + agents + prototypes"}
        elif name == "generate_creative_narrative":
            return f"Creative {args.get('style', 'lyrical')} narrative about {args.get('theme')}"
        elif name == "query_mesh_status":
            return {"status": "healthy", "component": args.get("component")}
        elif name == "suggest_qcoin_incentive":
            return {"incentive": f"QCoin for {args.get('action_type')}", "mechanism": "rune + mesh reputation"}
        else:
            return {"unknown_tool": name, "args": args}

    def agentic_react_loop(
        self, user_input: str, max_steps: int = 5, model: str = "grok-3"
    ) -> str:
        """Sync ReAct loop using real Grok tool calling."""
        messages = [
            {"role": "system", "content": f"You are {self.name} Nexus agent. Use tools when helpful."},
            {"role": "user", "content": user_input}
        ]

        for _ in range(max_steps):
            response = self.call_grok(messages, tools=self.tools, model=model)
            if "error" in response:
                return f"Error: {response['error']}"

            message = response["choices"][0]["message"]
            messages.append(message)

            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                return message.get("content", "No final answer")

            for tc in tool_calls:
                result = self.execute_tool_call(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(result)
                })

        return "Max steps reached."

    # ==================== ASYNC GROK METHODS (NEW) ====================

    async def async_call_grok(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """ASYNC Grok call with full tool support. Non-blocking."""
        if not self.async_client:
            # Fallback to sync simulated if async not available
            sync_result = self.call_grok(messages, tools, model, temperature, max_tokens, tool_choice)
            return sync_result

        tools = tools or self.tools
        try:
            resp = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.model_dump()
        except Exception as e:
            return {"error": str(e)}

    async def async_execute_tool_call(self, tool_call: Dict) -> Any:
        """Async version of tool execution. Override for real async operations (mesh, blockchain, I/O)."""
        # For now delegate to sync version (most tools are fast).
        # Future: make mesh queries, QCoin RPCs, hardware calls truly async here.
        return self.execute_tool_call(tool_call)

    async def async_agentic_react_loop(
        self, user_input: str, max_steps: int = 5, model: str = "grok-3"
    ) -> str:
        """ASYNC ReAct loop with real Grok tool calling. Ideal for concurrent swarms or I/O-heavy agents."""
        messages = [
            {"role": "system", "content": f"You are {self.name} Nexus agent. Use tools when helpful."},
            {"role": "user", "content": user_input}
        ]

        for _ in range(max_steps):
            response = await self.async_call_grok(messages, tools=self.tools, model=model)
            if "error" in response:
                return f"Error: {response['error']}"

            message = response["choices"][0]["message"]
            messages.append(message)

            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                return message.get("content", "No final answer from Grok")

            for tc in tool_calls:
                result = await self.async_execute_tool_call(tc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(result)
                })

        return "Max steps reached in async loop."

    # ==================== COMMON METHODS ====================

    @abstractmethod
    def embody(self, context: str = "") -> str:
        pass

    def handoff(self, target_agent: str, context: str) -> str:
        return f"HANDOFF::{target_agent}::{context[:200]}..."

    def log_event(self, event_type: str, details: str):
        events = self.get_state("events", [])
        events.append({"type": event_type, "details": details, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.update_state("events", events)

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        self.log_event("tool_use", f"{tool_name} with {kwargs}")
        return {"status": "prefer agentic_react_loop or async_agentic_react_loop", "tool": tool_name}

    def integrate_with_mesh(self, action: str):
        return self.use_tool("mesh", action=action)

    def integrate_with_blockchain(self, action: str):
        return self.use_tool("blockchain", action=action)

    def __repr__(self):
        grok_mode = "async+sync" if self.async_client else ("sync-only" if self.client else "simulated")
        return f"<{self.name}Agent sessions={self.get_state('session_count')} grok={grok_mode}>"
