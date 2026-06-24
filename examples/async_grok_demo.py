#!/usr/bin/env python3
"""
Async Grok Tool Calling Demo for Nexus Agent Swarm

Demonstrates the new async capabilities added to BaseAgent:
- async_agentic_react_loop()
- async_call_grok()

Run with a real XAI_API_KEY for full Grok intelligence + tool execution.
Without a key it gracefully falls back to simulated mode.

Usage:
    export XAI_API_KEY=sk-...
    python examples/async_grok_demo.py
"""

import asyncio
import os
import sys

from agents.nexus_orchestrator import NexusOrchestrator


async def main():
    print("=" * 60)
    print("NEXUS AGENT SWARM - ASYNC GROK TOOL CALLING DEMO")
    print("=" * 60)

    # Set your key (or it will run in simulated mode)
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("\n[WARNING] XAI_API_KEY not found in environment.")
        print("Running in SIMULATED mode. Set the key for real Grok calls.\n")
        os.environ["XAI_API_KEY"] = "simulated"  # triggers simulated path

    nexus = NexusOrchestrator()
    print(f"\nInstantiated: {nexus}")
    print(f"Swarm members: {nexus.get_state('swarm_members')}")

    print("\n" + "-" * 60)
    print("1. ASYNC ReAct LOOP with real Grok tool calling")
    print("-" * 60)

    user_input = (
        "Design QCoin incentives for a swarm of emotional (Lyra) + technical (Xen) agents "
        "and generate a short mythic narrative around it."
    )

    try:
        result = await nexus.async_agentic_react_loop(
            user_input=user_input,
            max_steps=4,
            model="grok-3"
        )
        print("\n[RESULT]\n", result)
    except Exception as e:
        print(f"Error in async_agentic_react_loop: {e}")

    print("\n" + "-" * 60)
    print("2. DIRECT async_call_grok with tools")
    print("-" * 60)

    try:
        response = await nexus.async_call_grok(
            messages=[
                {"role": "system", "content": "You are a helpful Nexus systems analyst."},
                {"role": "user", "content": "Analyze mesh + QNET integration risks and suggest mitigation strategies."}
            ],
            tools=nexus.tools,
            model="grok-3"
        )

        if "error" in response:
            print(f"Grok returned error: {response['error']}")
        else:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "No content")
            print("\n[GROK RESPONSE]\n", content)

            # Show if any tool calls were made
            tool_calls = response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
            if tool_calls:
                print(f"\n[Tool calls made]: {len(tool_calls)}")
    except Exception as e:
        print(f"Error in async_call_grok: {e}")

    print("\n" + "=" * 60)
    print("Demo complete. The swarm is ready for concurrent async operations.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
