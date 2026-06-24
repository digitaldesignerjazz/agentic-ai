#!/usr/bin/env python3
"""
Nexus Agent SSE Streaming Server

Production-ready Server-Sent Events (SSE) endpoint for real-time
streaming of Nexus agent responses powered by Grok.

Features:
- Async streaming via async_stream_grok()
- Proper SSE event format
- CORS enabled (for browser / frontend use)
- Query parameters for model and temperature
- Graceful simulated mode when no API key is set

Run:
    pip install fastapi uvicorn
    uvicorn examples.sse_agent_stream:app --reload --port 8000

Test:
    curl -N "http://localhost:8000/stream?prompt=Hello%20Nexus"
"""

import asyncio
import os
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents.nexus_orchestrator import NexusOrchestrator

app = FastAPI(
    title="Nexus Agent SSE Stream",
    description="Real-time streaming for Nexus AI Agent Swarm using Grok + SSE",
    version="0.1.0"
)

# Enable CORS for browser-based clients and Grok Launcher integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # In production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single orchestrator instance (stateless for now)
nexus = NexusOrchestrator()


async def sse_event_generator(
    prompt: str,
    model: str = "grok-3",
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    """
    Generates SSE-formatted events from async_stream_grok.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and insightful Nexus agent. "
                       "Respond clearly and concisely.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        async for chunk in nexus.async_stream_grok(
            messages=messages,
            model=model,
            temperature=temperature,
        ):
            if chunk:
                # Standard SSE format
                yield f"event: message\ndata: {chunk}\n\n"

        # Signal completion
        yield "event: done\ndata: [DONE]\n\n"

    except Exception as e:
        yield f"event: error\ndata: {str(e)}\n\n"


@app.get("/stream", tags=["Streaming"])
async def stream_agent(
    prompt: str = Query(..., min_length=1, description="Prompt to send to the agent"),
    model: str = Query("grok-3", description="Grok model to use"),
    temperature: float = Query(0.7, ge=0.0, le=2.0, description="Sampling temperature"),
):
    """
    Stream agent responses using Server-Sent Events (SSE).

    Returns a continuous stream of tokens in real time.
    """
    return StreamingResponse(
        sse_event_generator(prompt, model, temperature),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
        },
    )


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": str(nexus),
        "grok_streaming": "enabled",
        "sse": "active",
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("NEXUS AGENT SSE STREAMING SERVER")
    print("=" * 60)
    print("Starting server on http://0.0.0.0:8000")
    print("Test with: curl -N http://localhost:8000/stream?prompt=Hello")
    print("=" * 60 + "\n")

    uvicorn.run(
        "examples.sse_agent_stream:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
