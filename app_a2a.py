from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
from app_agent import run_agent
from dotenv import load_dotenv

app = FastAPI()

import os
load_dotenv()

# ---- A2A-ish schemas (trimmed for demo) ----
class A2AMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str

class A2ARequest(BaseModel):
    conversation: List[A2AMessage]
    context: Optional[dict] = None

class A2AResponse(BaseModel):
    messages: List[A2AMessage]
    # You can add "actions" / "tool_calls" fields if your agent calls tools

# Manifest tells Agentspace what your agent is and how to call it
@app.get("/.well-known/agent.json")
def manifest():
    return {
        "name": "lc-demo-agent",
        "description": "LangChain demo agent via A2A",
        "protocol": "a2a.v1",
        "entrypoints": {"messages": "/a2a/messages"},
        "capabilities": ["text"],
        "vendor": {"framework": "langchain"}
    }

# Core message bridge
@app.post("/a2a/messages", response_model=A2AResponse)
def a2a_messages(req: A2ARequest):
    # Simple policy: take the latest user message
    latest_user = next((m for m in reversed(req.conversation) if m.role == "user"), None)
    prompt = latest_user.content if latest_user else "Hello!"
    answer = run_agent(prompt)
    return A2AResponse(messages=[A2AMessage(role="assistant", content=answer)])
