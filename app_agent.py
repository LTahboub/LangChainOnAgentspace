# app_agent.py
import os
from langchain_core.messages import HumanMessage

_llm = None
def _llm():
    # lazy init to avoid crashing on import
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

def run_agent(prompt: str) -> str:
    try:
        resp = _llm().invoke([HumanMessage(content=prompt)])
        return resp.content
    except Exception as e:
        # Never hard-crash Cloud Run on missing creds: degrade instead
        return f"[fallback] {prompt}"
