#!/usr/bin/env python
"""Example LangChain server exposes multiple runnables (LLMs in this case)."""
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI


from langserve import add_routes

from langgraph.prebuilt import chat_agent_executor


from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolExecutor
## Define the input schema for the tool function. In this case the tool accepts a query string as input.

class Wikipeida_tool_arg_schema(BaseModel):
    """Input for the Wikipeida tool."""

    query: str = Field(description="search query to look up")

# A list of tools are provided for the agent to utilize
tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(), args_schema=Wikipeida_tool_arg_schema)]

# We will set streaming=True so that we can stream tokens
model = ChatOpenAI(temperature=0, streaming=True)

# Create the chat agent using LangGraph built-in function
agent = chat_agent_executor.create_function_calling_executor(model, tools)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    agent,
    path="/openai",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)