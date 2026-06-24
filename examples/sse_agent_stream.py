#!/usr/bin/env python3
"""
Nexus Agent SSE Server with Advanced LangGraph ReAct Streaming

Uses LangGraph + astream_events() for rich, real-time streaming of:
- Reasoning steps
- Tool calls
- Tool results
- Final answer
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

app = FastAPI(title="Nexus Agent SSE + LangGraph ReAct (Advanced Streaming)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def advanced_react_stream(
    prompt: str,
    model: str = "grok-3",
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    if not LANGGRAPH_AVAILABLE:
        yield "event: error\ndata: LangGraph not available. Install with: pip install langgraph langchain langchain-openai\n\n"
        return

    try:
        agent = create_nexus_react_agent(model=model, temperature=temperature)
        inputs = {"messages": [("user", prompt)]}
        config = {"configurable": {"thread_id": "sse-react-session"}}

        yield f"event: start\ndata: {json.dumps({'prompt': prompt})}\n\n"

        # Use astream_events for granular streaming
        async for event in agent.astream_events(inputs, config=config, version="v2"):
            kind = event.get("event")
            data = event.get("data", {})

            if kind == "on_chat_model_stream":
                # Streaming tokens from the LLM
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield f"event: message\ndata: {chunk.content}\n\n"

            elif kind == "on_tool_start":
                tool_name = data.get("name", "unknown_tool")
                tool_input = data.get("input", {})
                yield f"event: tool_call\ndata: {json.dumps({'tool': tool_name, 'input': tool_input})}\n\n"

            elif kind == "on_tool_end":
                tool_name = data.get("name", "unknown_tool")
                output = data.get("output", {})
                yield f"event: tool_result\ndata: {json.dumps({'tool': tool_name, 'output': output})}\n\n"
            elif kind == "on_chain_end":
                # Final output of the agent
                outputs = data.get("output", {})
                if "messages" in outputs:
                    final_msg = outputs["messages"][-1]
                    if hasattr(final_msg, "content"):
                        yield f"event: final_answer\ndata: {final_msg.content}\n\n"
        yield "event: done\ndata: [DONE]\n\n"
    except Exception as e:
        yield f"event: error\ndata: {str(e)}\n\n"


@app.get("/react")
async def react_endpoint(
    prompt: str = Query(...),
    model: str = Query("grok-3"),
    temperature: float = Query(0.7),
):
    """Full ReAct loop with rich SSE streaming (reasoning + tool calls)."""
    return StreamingResponse(
        advanced_react_stream(prompt, model, temperature),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "langgraph": LANGGRAPH_AVAILABLE,
        "streaming": "advanced (astream_events)",
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting advanced Nexus SSE Server with LangGraph ReAct streaming...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
