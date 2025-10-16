"""
Completions API Route
POST /v1/completions - Text completion with streaming
"""

import time
import uuid
import json
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import TYPE_CHECKING

import openvino_genai as ov_genai

from ..models import CompletionRequest, CompletionResponse, CompletionChoice, Usage

if TYPE_CHECKING:
    from ..managers import ModelManager

router = APIRouter()
model_manager: 'ModelManager' = None


def set_model_manager(manager):
    global model_manager
    model_manager = manager


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Create text completion (OpenAI-compatible)"""
    
    pipeline = model_manager.get_pipeline(request.model)
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = request.max_tokens
    config.temperature = request.temperature
    config.top_p = request.top_p
    config.do_sample = request.temperature > 0
    
    if request.stream:
        return StreamingResponse(
            stream_completion(pipeline, prompt, request, config),
            media_type="text/event-stream"
        )
    else:
        response_text = pipeline.generate(prompt, config)
        
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[CompletionChoice(
                index=0,
                text=response_text,
                finish_reason="stop"
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )


async def stream_completion(
    pipeline: ov_genai.LLMPipeline,
    prompt: str,
    request: CompletionRequest,
    config: ov_genai.GenerationConfig
):
    """Stream text completion responses"""
    chunk_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    class TokenStreamer:
        def __init__(self):
            self.tokens = []
        
        def __call__(self, token: str) -> int:
            self.tokens.append(token)
            return 0
    
    streamer = TokenStreamer()
    result = pipeline.generate(prompt, config, streamer=streamer)
    
    for token in streamer.tokens:
        chunk = {
            "id": chunk_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "text": token, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)
    
    final_chunk = {
        "id": chunk_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

