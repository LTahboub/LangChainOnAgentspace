from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import os
load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini")  # or your preferred model/provider
def run_agent(prompt: str) -> str:
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content
