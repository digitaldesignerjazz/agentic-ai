#!/usr/bin/env python3
"""
Nexus Agent SSE Server with LangGraph ReAct Support

This version uses LangGraph for a more robust ReAct implementation.
"""

import asyncio
import json
import os
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

try:
    from core.langgraph_react import create_nexus_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

app = FastAPI(title="Nexus Agent SSE + LangGraph ReAct")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def langgraph_react_stream(
    prompt: str,
    model: str = "grok-3",
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    if not LANGGRAPH_AVAILABLE:
        yield "event: error\ndata: LangGraph not installed. Run: pip install langgraph langchain langchain-openai\n\n"
        return

    try:
        agent = create_nexus_react_agent(model=model, temperature=temperature)
        inputs = {"messages": [("user", prompt)]}
        config = {"configurable": {"thread_id": "sse-session"}}

        yield f"event: start\ndata: {json.dumps({'prompt': prompt})}\n\n"

        async for event in agent.astream(inputs, config=config):
            # LangGraph emits different event types
            for node, data in event.items():
                if "messages" in data:
                    last_msg = data["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        yield f"event: message\ndata: {last_msg.content}\n\n"

        yield "event: done\ndata: [DONE]\n\n"

    except Exception as e:
        yield f"event: error\ndata: {str(e)}\n\n"


@app.get("/react")
async def react_endpoint(
    prompt: str = Query(...),
    model: str = Query("grok-3"),
    temperature: float = Query(0.7),
):
    """Run full ReAct loop using LangGraph over SSE."""
    return StreamingResponse(
        langgraph_react_stream(prompt, model, temperature),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "langgraph": LANGGRAPH_AVAILABLE,
        "endpoints": ["/react"],
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Nexus SSE Server with LangGraph ReAct...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
