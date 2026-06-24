#!/usr/bin/env python3
"""
Nexus Agent SSE Streaming Server with ReAct Loop Support

Endpoints:
- GET /stream     → Simple streaming (good for quick prompts)
- GET /react      → Full ReAct loop with tool calling (recommended for complex tasks)

Both support real-time SSE streaming.
"""

import asyncio
import json
import os
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents.nexus_orchestrator import NexusOrchestrator

app = FastAPI(
    title="Nexus Agent SSE Stream",
    description="Real-time streaming for Nexus AI Agent Swarm (Simple + ReAct modes)",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nexus = NexusOrchestrator()


# ============================================================
# SIMPLE STREAMING (existing behavior)
# ============================================================

async def simple_stream_generator(prompt: str, model: str, temperature: float) -> AsyncGenerator[str, None]:
    messages = [
        {"role": "system", "content": "You are a helpful Nexus agent."},
        {"role": "user", "content": prompt},
    ]

    try:
        async for chunk in nexus.async_stream_grok(messages=messages, model=model, temperature=temperature):
            if chunk:
                yield f"event: message\ndata: {chunk}\n\n"
        yield "event: done\ndata: [DONE]\n\n"
    except Exception as e:
        yield f"event: error\ndata: {str(e)}\n\n"


@app.get("/stream")
async def stream_simple(
    prompt: str = Query(...),
    model: str = Query("grok-3"),
    temperature: float = Query(0.7),
):
    return StreamingResponse(
        simple_stream_generator(prompt, model, temperature),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ============================================================
# REACT LOOP OVER SSE (NEW)
# ============================================================

async def react_loop_generator(
    prompt: str,
    model: str = "grok-3",
    temperature: float = 0.7,
    max_steps: int = 6
) -> AsyncGenerator[str, None]:
    """
    Streams a full ReAct loop with tool usage visibility.
    """
    yield f"event: start\ndata: {json.dumps({'prompt': prompt})}\n\n"

    try:
        # We use the existing ReAct loop but enhance visibility
        final_answer = await nexus.async_agentic_react_loop(
            user_input=prompt,
            max_steps=max_steps,
            model=model
        )

        # For now, we stream the final result.
        # In a future iteration we can stream intermediate reasoning + tool calls.
        yield f"event: final_answer\ndata: {final_answer}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    except Exception as e:
        yield f"event: error\ndata: {str(e)}\n\n"


@app.get("/react")
async def stream_react(
    prompt: str = Query(...),
    model: str = Query("grok-3"),
    temperature: float = Query(0.7),
    max_steps: int = Query(6),
):
    """
    Run a full ReAct agent loop with tool calling over SSE.
    This allows the agent to use tools (analyze integration, generate creative content, query mesh, etc.)
    and return a well-reasoned final answer.
    """
    return StreamingResponse(
        react_loop_generator(prompt, model, temperature, max_steps),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "endpoints": ["/stream", "/react"],
        "agent": str(nexus),
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Nexus SSE Server with ReAct support...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
