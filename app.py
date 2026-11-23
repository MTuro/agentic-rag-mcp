"""ASGI entry point for the local HTTP interface."""

from fastapi import FastAPI

from api.routes import router


app = FastAPI(title="Agentic RAG MCP")
app.include_router(router)
