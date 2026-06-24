#!/usr/bin/env python3
"""
SSE (Server-Sent Events) Streaming for Nexus Agent Swarm

This example exposes the async_stream_grok() capability over HTTP using SSE.

Useful for:
- Web frontends (Grok Launcher, Nexus Control Panel, custom dashboards)
- Real-time UI updates
- Integration with other services in the Nexus ecosystem

Run with:
    pip install fastapi uvicorn
    uvicorn examples.sse_agent_stream:app --reload

Then open in browser or use curl:
    curl http://localhost:8000/stream?prompt=Hello%20Nexus
"""

import asyncio
import os
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from agents.nexus_orchestrator import NexusOrchestrator

app = FastAPI(title="Nexus Agent SSE Stream")

# Global orchestrator instance (in production you might want per-session instances)
nexus = NexusOrchestrator()


async def generate_sse_stream(prompt: str) -> AsyncGenerator[str, None]:
    """
    Convert async_stream_grok into proper SSE format.
    Each chunk is sent as: data: {chunk}\n\n
    """
    messages = [
        {"role": "system", "content": "You are a helpful Nexus agent. Be concise and insightful."},
        {"role": "user", "content": prompt}
    ]

    try:
        async for chunk in nexus.async_stream_grok(
            messages=messages,
            model="grok-3",
            temperature=0.7
        ):
            if chunk:
                # SSE format: data: message\n\n
                yield f"data: {chunk}\n\n"

        # Optional: send a final [DONE] event
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: [ERROR] {str(e)}\n\n"


@app.get("/stream")
async def stream_agent_response(
    prompt: str = Query(..., description="The prompt to send to the Nexus agent")
):
    """
    Stream agent responses using Server-Sent Events (SSE).
    """
    return StreamingResponse(
        generate_sse_stream(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Important for nginx/proxies
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent": str(nexus),
        "streaming": "enabled via async_stream_grok"
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Nexus SSE Streaming Server...")
    print("Try: curl http://localhost:8000/stream?prompt=Explain%20QCoin")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
