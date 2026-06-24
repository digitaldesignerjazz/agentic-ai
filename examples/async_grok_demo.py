#!/usr/bin/env python3
"""
Async + Streaming Grok Tool Calling Demo for Nexus Agent Swarm

Demonstrates:
- async_agentic_react_loop()
- async_call_grok()
- NEW: async_stream_grok()  (token-by-token real-time output)

Run with a real XAI_API_KEY for the best experience.
Without a key it uses nice simulated streaming.
"""

import asyncio
import os
import sys

from agents.nexus_orchestrator import NexusOrchestrator


async def main():
    print("=" * 65)
    print("NEXUS AGENT SWARM - ASYNC + STREAMING GROK DEMO")
    print("=" * 65)

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("\n[INFO] No XAI_API_KEY found → Running in SIMULATED STREAMING mode.\n")

    nexus = NexusOrchestrator()
    print(f"\nInstantiated: {nexus}")
    print(f"Swarm members: {nexus.get_state('swarm_members')}")

    # ============================================================
    # 1. ASYNC ReAct LOOP (still non-streaming for tool-using agents)
    # ============================================================
    print("\n" + "-" * 65)
    print("1. ASYNC ReAct LOOP with tool calling")
    print("-" * 65)

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
        print("\n[ReAct RESULT]\n", result)
    except Exception as e:
        print(f"Error: {e}")

    # ============================================================
    # 2. STREAMING OUTPUT (NEW - token by token)
    # ============================================================
    print("\n" + "-" * 65)
    print("2. STREAMING OUTPUT with async_stream_grok()  ← Real-time tokens")
    print("-" * 65)

    try:
        print("\n[STREAMING RESPONSE]\n", end="", flush=True)

        async for chunk in nexus.async_stream_grok(
            messages=[
                {"role": "system", "content": "You are a helpful and poetic Nexus systems analyst."},
                {"role": "user", "content": "Write a short, beautiful paragraph about the emergence of sovereign AI agent swarms on a decentralized mesh network."}
            ],
            tools=nexus.tools,           # still available if the model decides to use tools
            model="grok-3",
            temperature=0.85
        ):
            print(chunk, end="", flush=True)

        print("\n")  # final newline
    except Exception as e:
        print(f"\nStreaming error: {e}")

    print("\n" + "=" * 65)
    print("Demo complete. Streaming makes long responses feel alive.")
    print("=" * 65)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
