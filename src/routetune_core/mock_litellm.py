from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MOCK_MODELS = {
    "mock-fast": "简短、低成本的本地模拟输出",
    "mock-balanced": "中等详细度的本地模拟输出",
    "mock-strong": "更完整的本地模拟输出",
}


class MockMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str


class MockCompletionInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[MockMessage] = Field(min_length=1)
    stream: bool = False


def create_mock_litellm_app() -> FastAPI:
    app = FastAPI(title="Local LiteLLM-compatible Mock", version="0.1.0")

    @app.get("/health")
    @app.get("/health/liveliness")
    @app.get("/health/readiness")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "owned_by": "local-litellm-mock"}
                for model_id in MOCK_MODELS
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completion(payload: MockCompletionInput) -> dict[str, Any]:
        if payload.stream:
            raise HTTPException(status_code=400, detail="mock streaming is not supported")
        if payload.model not in MOCK_MODELS:
            raise HTTPException(status_code=404, detail=f"unknown mock model: {payload.model}")
        user_text = next(
            (message.content for message in reversed(payload.messages) if message.role == "user"),
            payload.messages[-1].content,
        )
        content = _mock_answer(payload.model, user_text)
        prompt_tokens = max(1, len(user_text) // 4)
        completion_tokens = max(1, len(content) // 4)
        return {
            "id": f"mock-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def _mock_answer(model: str, user_text: str) -> str:
    normalized = " ".join(user_text.split())
    if model == "mock-fast":
        return f"[LiteLLM mock · fast] 已接收请求：{normalized[:80]}"
    if model == "mock-balanced":
        return (
            "[LiteLLM mock · balanced] 这是用于验证 RouteTune 路由、Trace 与标注链路的"
            f"确定性响应。请求摘要：{normalized[:160]}"
        )
    return (
        "[LiteLLM mock · strong] 这是本地强模型占位输出，用于与平级外部 Provider 做批次"
        f"对比和人工标注。输入内容：{normalized[:240]}"
    )
