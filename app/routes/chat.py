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
    ChatCompletionStreamChoice,
    ContentLogprobs,
    TokenLogprob
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
    
    # Log request
    print(f"📨 Chat: model={request.model}, msgs={len(request.messages)}, stream={request.stream}, modalities={request.modalities}")
    
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
            # OpenVINO GenAI expects a set, not a list
            config.stop_strings = set(stop_strings)
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
    
    # VLM history management: Reload on error
    # When VLM accumulates too much history, it throws:
    # "Check 'prompt_ids.get_size() >= tokenized_history.size()' failed"
    
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
        error_msg = str(e)
        # Check if it's a VLM error that requires pipeline reload
        vlm_reload_errors = [
            "prompt_ids.get_size() >= tokenized_history.size()",
            "Prompt ids size is less than tokenized history size",
            "MAX_PROMPT_LEN",  # NPU prompt length exceeded
            "inputs_embeds.get__shape().at(1) <= m_max_prompt_len"  # NPU max prompt check
        ]
        if model_type == "vlm" and any(err_pattern in error_msg for err_pattern in vlm_reload_errors):
            print(f"⚠️  VLM error detected, reloading pipeline with updated config...")
            # Get model manager to reload the VLM
            from ..main import model_manager
            if model_manager and model_manager.reload_vlm_pipeline(request.model):
                # Retry after reload
                print(f"🔄 Retrying generation after VLM reload...")
                pipeline = model_manager.get_pipeline(request.model)
                try:
                    if images:
                        image_tensor = images[0]
                        result = pipeline.generate(prompt, image=image_tensor, max_new_tokens=config.max_new_tokens)
                        response_text = result if isinstance(result, str) else result.texts[0]
                    else:
                        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
                        dummy_tensor = ov.Tensor(dummy_image)
                        result = pipeline.generate(prompt, image=dummy_tensor, max_new_tokens=config.max_new_tokens)
                        response_text = result if isinstance(result, str) else result.texts[0]
                except Exception as retry_error:
                    raise HTTPException(status_code=500, detail=f"Generation failed after reload: {str(retry_error)}")
            else:
                raise HTTPException(status_code=500, detail=f"VLM reload failed: {error_msg}")
        else:
            raise HTTPException(status_code=500, detail=f"Generation failed: {error_msg}")
    
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
    
    # Generate logprobs if requested
    logprobs_data = None
    if request.logprobs:
        # Note: OpenVINO GenAI doesn't expose token-level probabilities via standard API
        # This is a basic implementation that returns token-level structure
        # For true logprobs, would need access to model's internal token probabilities
        tokens = (final_content or response_text).split()
        logprobs_list = []
        for token in tokens:
            # Placeholder logprob values (would need actual model outputs)
            logprobs_list.append(TokenLogprob(
                token=token,
                logprob=-0.5,  # Placeholder
                bytes=list(token.encode('utf-8')),
                top_logprobs=None if not request.top_logprobs else [
                    {"token": token, "logprob": -0.5}
                ]
            ))
        logprobs_data = ContentLogprobs(content=logprobs_list)
    
    # Generate audio if modalities includes "audio" (GPT-4o style)
    if request.modalities and "audio" in request.modalities:
        from ..models import AudioData
        from ..main import model_manager
        import base64
        
        tts_models = list(model_manager.tts_pipelines.keys())
        if tts_models:
            try:
                tts_pipeline = model_manager.tts_pipelines[tts_models[0]]
                text_to_speak = final_content or response_text
                
                # Generate audio
                audio_result = tts_pipeline.generate(text_to_speak)
                audio_array = audio_result
                
                # Convert to base64
                import io
                import soundfile as sf
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio_array, 16000, format='WAV')
                audio_buffer.seek(0)
                audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
                
                # Add audio to response
                response_message.audio = AudioData(
                    id=f"audio_{uuid.uuid4().hex[:16]}",
                    data=audio_base64,
                    transcript=text_to_speak
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not generate audio: {e}")
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(
            index=0,
            message=response_message,
            finish_reason=finish_reason,
            logprobs=logprobs_data
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

