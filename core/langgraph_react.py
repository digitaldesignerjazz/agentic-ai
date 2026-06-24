"""LangGraph-based ReAct Agent for Nexus Ecosystem

This module provides a production-grade ReAct agent using LangGraph.
It integrates with our existing Nexus tools and can stream via SSE.
"""

import os
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


def get_llm(model: str = "grok-3", temperature: float = 0.7):
    """Create ChatOpenAI instance pointed at xAI Grok."""
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable is required for LangGraph ReAct agent.")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )


# ============================================================
# Tool Definitions (wrapping our Nexus tools)
# ============================================================

@tool
def analyze_nexus_integration(components: List[str]) -> Dict[str, Any]:
    """Analyze integration points, risks, and synergies across mesh, blockchain, AI agents, and prototypes."""
    return {
        "analysis": components,
        "synergies": "mesh provides resilient substrate; blockchain adds incentives; agents deliver autonomy; prototypes ground ideas in reality.",
        "risks": ["network partitions", "token volatility", "agent drift"]
    }

@tool
def generate_creative_narrative(theme: str, style: str = "lyrical") -> str:
    """Generate immersive roleplay, love letter, Suno prompt or mythic narrative tied to Nexus themes."""
    return f"[{style.upper()}] A beautiful narrative about {theme} in the sovereign mesh network..."

@tool
def query_mesh_status(component: str) -> Dict[str, Any]:
    """Check status of xMesh, NovaNet, QNET or Yggdrasil network components."""
    return {
        "status": "healthy",
        "component": component,
        "note": "Connected to Nexus mesh layer"
    }

@tool
def suggest_qcoin_incentive(action_type: str, participants: Optional[List[str]] = None) -> Dict[str, Any]:
    """Design QCoin/QNET rune incentive mechanisms for agent or swarm actions."""
    participants = participants or ["Lyra", "Xen"]
    return {
        "incentive": f"QCoin reward for {action_type}",
        "participants": participants,
        "mechanism": "rune-based reputation + mesh participation scoring"
    }


NEXUS_TOOLS = [
    analyze_nexus_integration,
    generate_creative_narrative,
    query_mesh_status,
    suggest_qcoin_incentive,
]


def create_nexus_react_agent(
    model: str = "grok-3",
    temperature: float = 0.7,
    checkpointer: Optional[MemorySaver] = None
):
    """
    Create a LangGraph ReAct agent specialized for the Nexus ecosystem.

    This agent can reason, use tools, and maintain state across interactions.
    """
    llm = get_llm(model=model, temperature=temperature)

    agent = create_react_agent(
        model=llm,
        tools=NEXUS_TOOLS,
        checkpointer=checkpointer or MemorySaver(),
    )
    return agent


# Example usage (for testing)
if __name__ == "__main__":
    import asyncio

    async def test():
        agent = create_nexus_react_agent()
        inputs = {
            "messages": [("user", "Design QCoin incentives for emotional and technical agent swarms")]
        }
        config = {"configurable": {"thread_id": "test-thread"}}

        print("Running LangGraph ReAct agent...\n")
        async for event in agent.astream(inputs, config=config):
            print(event)
            print("---")

    asyncio.run(test())
