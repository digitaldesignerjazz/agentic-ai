#!/usr/bin/env python3
"""
Nexus Agent WebSocket Server

Real-time bidirectional communication with Nexus agents using WebSockets.

Advantages over SSE:
- Bidirectional (client can send commands mid-stream)
- Lower latency for interactive sessions
- Better for complex agent interactions

Run:
    pip install fastapi uvicorn
    uvicorn examples.websocket_agent:app --reload --port 8001

Client example (JavaScript):
    const ws = new WebSocket("ws://localhost:8001/ws");
    ws.onmessage = (event) => console.log(JSON.parse(event.data));
    ws.send(JSON.stringify({type: "prompt", prompt: "Hello Nexus"}));
"""

import asyncio
import json
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    from core.langgraph_react import create_nexus_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

app = FastAPI(title="Nexus Agent WebSocket Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_text(json.dumps(data))


manager = ConnectionManager()


async def stream_langgraph_to_websocket(
    websocket: WebSocket,
    prompt: str,
    model: str = "grok-3",
    temperature: float = 0.7
):
    """Stream LangGraph ReAct agent output over WebSocket."""
    if not LANGGRAPH_AVAILABLE:
        await manager.send_json(websocket, {
            "type": "error",
            "message": "LangGraph not installed"
        })
        return

    try:
        agent = create_nexus_react_agent(model=model, temperature=temperature)
        inputs = {"messages": [("user", prompt)]}
        config = {"configurable": {"thread_id": "ws-session"}}

        await manager.send_json(websocket, {
            "type": "start",
            "prompt": prompt
        })

        async for event in agent.astream_events(inputs, config=config, version="v2"):
            kind = event.get("event")
            data = event.get("data", {})

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    await manager.send_json(websocket, {
                        "type": "message",
                        "content": chunk.content
                    })

            elif kind == "on_tool_start":
                await manager.send_json(websocket, {
                    "type": "tool_call",
                    "tool": data.get("name"),
                    "input": data.get("input", {})
                })

            elif kind == "on_tool_end":
                await manager.send_json(websocket, {
                    "type": "tool_result",
                    "tool": data.get("name"),
                    "output": data.get("output", {})
                })

            elif kind == "on_chain_end":
                outputs = data.get("output", {})
                if "messages" in outputs:
                    final_msg = outputs["messages"][-1]
                    if hasattr(final_msg, "content"):
                        await manager.send_json(websocket, {
                            "type": "final_answer",
                            "content": final_msg.content
                        })

        await manager.send_json(websocket, {"type": "done"})

    except Exception as e:
        await manager.send_json(websocket, {
            "type": "error",
            "message": str(e)
        })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": "Invalid JSON"
                })
                continue

            msg_type = message.get("type")

            if msg_type == "prompt":
                prompt = message.get("prompt", "")
                model = message.get("model", "grok-3")
                temperature = message.get("temperature", 0.7)

                if prompt:
                    await stream_langgraph_to_websocket(
                        websocket, prompt, model, temperature
                    )
                else:
                    await manager.send_json(websocket, {
                        "type": "error",
                        "message": "Prompt is required"
                    })

            elif msg_type == "ping":
                await manager.send_json(websocket, {"type": "pong"})

            else:
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_json(websocket, {
            "type": "error",
            "message": str(e)
        })
        manager.disconnect(websocket)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "websocket": "/ws",
        "langgraph": LANGGRAPH_AVAILABLE,
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Nexus WebSocket Server...")
    print("Connect to: ws://localhost:8001/ws")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
