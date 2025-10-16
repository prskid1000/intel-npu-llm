"""
Chat Completions API Route
POST /v1/chat/completions - Chat with streaming, tools, JSON mode support
"""

import time
import uuid
import json
import asyncio
import numpy as np
from typing import Union, List, Dict, Any, Optional, TYPE_CHECKING
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import openvino_genai as ov_genai
import openvino as ov

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    Usage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice
)
from ..utils import (
    build_chat_prompt,
    parse_tool_calls_from_response,
    validate_json_schema,
    extract_json_from_response,
    extract_content_parts
)

if TYPE_CHECKING:
    from ..managers import ModelManager, FileStorageManager, DocumentProcessor

router = APIRouter()

# Global references (set by main.py)
model_manager: 'ModelManager' = None
file_storage: 'FileStorageManager' = None
document_processor = None


def set_dependencies(mm, fs, dp):
    """Set manager dependencies"""
    global model_manager, file_storage, document_processor
    model_manager = mm
    file_storage = fs
    document_processor = dp


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Create chat completion (OpenAI-compatible) with multimodal, tool calling, and structured output support"""
    
    pipeline = model_manager.get_pipeline(request.model)
    model_type = model_manager.get_model_type(request.model)
    
    # Configure generation
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = request.max_tokens
    config.temperature = request.temperature
    config.top_p = request.top_p
    config.do_sample = request.temperature > 0
    
    # Set seed for reproducibility
    if request.seed is not None:
        try:
            config.rng_seed = request.seed
        except AttributeError:
            pass
    
    # Handle stop sequences
    if request.stop:
        stop_strings = [request.stop] if isinstance(request.stop, str) else request.stop
        try:
            config.stop_strings = stop_strings
        except AttributeError:
            pass
    
    # Determine special modes
    use_tools = request.tools is not None and len(request.tools) > 0
    use_json_mode = False
    json_schema = None
    
    if request.response_format:
        if isinstance(request.response_format, dict):
            format_type = request.response_format.get("type", "text")
            if format_type == "json_object":
                use_json_mode = True
            elif format_type == "json_schema":
                use_json_mode = True
                json_schema = request.response_format.get("json_schema", {})
        elif hasattr(request.response_format, "type"):
            if request.response_format.type == "json_object":
                use_json_mode = True
            elif request.response_format.type == "json_schema":
                use_json_mode = True
                json_schema = request.response_format.json_schema
    
    # Build prompt
    if model_type == "vlm":
        prompt, images, file_ids = extract_content_parts(request.messages, file_storage)
    else:
        images = []
        file_ids = []
        prompt = build_chat_prompt(request.messages, request.tools if use_tools else None)
        
        if use_json_mode:
            json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only. "
            if json_schema:
                json_instruction += f"Follow this schema: {json.dumps(json_schema)}"
            else:
                json_instruction += "Do not include any text outside the JSON object."
            prompt = prompt.replace("<|im_start|>assistant\n", f"{json_instruction}\n<|im_start|>assistant\n")
    
    # Process RAG context
    rag_context = ""
    if file_ids:
        for file_id in file_ids:
            file_path = file_storage.get_file_path(file_id)
            if file_path:
                try:
                    from ..managers import DocumentProcessor
                    extracted_text = DocumentProcessor.extract_text(file_path)
                    if extracted_text and not extracted_text.startswith("["):
                        rag_context += f"\n\n---Document Content ({file_path.name})---\n{extracted_text}\n"
                except Exception as e:
                    print(f"Error extracting text from {file_id}: {e}")
    
    if rag_context:
        prompt = rag_context + "\n\n" + prompt
    
    # Generate response
    if request.stream:
        if model_type == "vlm":
            # VLM doesn't support streaming well, fall back to non-streaming
            return await chat_completions_non_streaming(
                pipeline, model_type, prompt, images, request, config,
                use_tools, use_json_mode, json_schema
            )
        else:
            return StreamingResponse(
                stream_chat_completion(pipeline, prompt, request, config),
                media_type="text/event-stream"
            )
    else:
        return await chat_completions_non_streaming(
            pipeline, model_type, prompt, images, request, config,
            use_tools, use_json_mode, json_schema
        )


async def chat_completions_non_streaming(
    pipeline: Union[ov_genai.LLMPipeline, ov_genai.VLMPipeline],
    model_type: str,
    prompt: str,
    images: List[ov.Tensor],
    request: ChatCompletionRequest,
    config: ov_genai.GenerationConfig,
    use_tools: bool = False,
    use_json_mode: bool = False,
    json_schema: Optional[Dict[str, Any]] = None
) -> ChatCompletionResponse:
    """Non-streaming chat completion handler"""
    
    try:
        if model_type == "vlm":
            if images:
                # VLM with images
                image_tensor = images[0]
                result = pipeline.generate(prompt, image=image_tensor, max_new_tokens=config.max_new_tokens)
                response_text = result if isinstance(result, str) else result.texts[0]
            else:
                # VLM text-only mode - create dummy image for NPU compatibility
                # Create a small 224x224 black image (minimal memory footprint)
                dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
                dummy_tensor = ov.Tensor(dummy_image)
                
                # Pass dummy image to satisfy NPU requirement
                result = pipeline.generate(prompt, image=dummy_tensor, max_new_tokens=config.max_new_tokens)
                response_text = result if isinstance(result, str) else result.texts[0]
        else:
            # LLM mode
            response_text = pipeline.generate(prompt, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    # Parse response
    tool_calls = None
    finish_reason = "stop"
    final_content = response_text
    
    if use_tools:
        cleaned_text, parsed_tool_calls = parse_tool_calls_from_response(response_text)
        if parsed_tool_calls:
            tool_calls = parsed_tool_calls
            final_content = cleaned_text if cleaned_text else None
            finish_reason = "tool_calls"
    
    if use_json_mode:
        if json_schema:
            is_valid, error_msg = validate_json_schema(response_text, json_schema)
            if not is_valid:
                json_content = extract_json_from_response(response_text)
                if json_content:
                    final_content = json_content
                else:
                    raise HTTPException(status_code=400, detail=f"Response does not match schema: {error_msg}")
            else:
                final_content = extract_json_from_response(response_text) or response_text
        else:
            json_content = extract_json_from_response(response_text)
            if json_content:
                final_content = json_content
    
    # Token estimation
    prompt_tokens = len(prompt.split())
    completion_tokens = len(response_text.split())
    
    response_message = ChatMessage(role="assistant", content=final_content)
    if tool_calls:
        response_message.tool_calls = tool_calls
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(
            index=0,
            message=response_message,
            finish_reason=finish_reason,
            logprobs=None
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        ),
        system_fingerprint=f"fp_{uuid.uuid4().hex[:16]}"
    )


async def stream_chat_completion(
    pipeline: ov_genai.LLMPipeline,
    prompt: str,
    request: ChatCompletionRequest,
    config: ov_genai.GenerationConfig
):
    """Stream chat completion responses"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    initial_chunk = ChatCompletionStreamResponse(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[ChatCompletionStreamChoice(
            index=0,
            delta={"role": "assistant", "content": ""},
            finish_reason=None
        )]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"
    
    try:
        try:
            for token in pipeline.generate(prompt, config, streamer=True):
                chunk = ChatCompletionStreamResponse(
                    id=chunk_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta={"content": token},
                        finish_reason=None
                    )]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0)
        except TypeError:
            result = pipeline.generate(prompt, config)
            words = result.split()
            for i, word in enumerate(words):
                content = word if i == 0 else f" {word}"
                chunk = ChatCompletionStreamResponse(
                    id=chunk_id,
                    created=created,
                    model=request.model,
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta={"content": content},
                        finish_reason=None
                    )]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.01)
        
        final_chunk = ChatCompletionStreamResponse(
            id=chunk_id,
            created=created,
            model=request.model,
            choices=[ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": "generation_failed"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

