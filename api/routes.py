"""Validate chat input and return the agent's result as JSON."""

from fastapi import APIRouter
from pydantic import BaseModel, StrictStr, field_validator

from agent import run_agent_with_result


router = APIRouter()


class ChatRequest(BaseModel):
    message: StrictStr

    @field_validator("message")
    @classmethod
    def require_non_blank_message(cls, message: str) -> str:
        if not message.strip():
            raise ValueError("Message must contain non-whitespace text.")
        return message


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    result = await run_agent_with_result(request.message)
    return ChatResponse(answer=result.answer, tools_used=result.tools_used)
