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
    extract_content_parts,
    apply_stop_sequences
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
        tools_to_include = request.tools if use_tools else None
        if tools_to_include:
            print(f"🔧 Including {len(tools_to_include)} tool(s) in prompt:")
            for tool in tools_to_include:
                print(f"   - {tool.function.name}: {tool.function.description}")
        prompt = build_chat_prompt(request.messages, tools_to_include, request.model)
        
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
    """Non-streaming chat completion handler with tool calling support"""
    
    # Build conversation history for tool calling (like Omni)
    conversation_messages = [msg.dict() for msg in request.messages] if request.messages else []
    max_iterations = 5  # Limit tool calling iterations
    iteration = 0
    final_response = None
    final_content = None
    tool_calls = None
    finish_reason = "stop"
    
    # Track the original prompt for first iteration
    original_prompt = prompt
    original_images = images
    
    # Tool calling loop (like Omni)
    while iteration < max_iterations:
        iteration += 1
        print(f"🔄 Tool calling iteration {iteration}/{max_iterations}")
        
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
        
        # Apply stop sequences if configured (post-processing workaround for OpenVINO GenAI)
        was_stopped = False
        if request.stop:
            stop_strings = [request.stop] if isinstance(request.stop, str) else request.stop
            print(f"🛑 Checking stop sequences: {stop_strings}")
            print(f"   Response length before: {len(response_text)} chars")
            response_text, was_stopped = apply_stop_sequences(response_text, stop_strings)
            if was_stopped:
                print(f"   ✅ Stop sequence triggered, truncated to {len(response_text)} chars")
            else:
                print(f"   ⚠️  No stop sequence found in response")
        
        # Store final response (will be overwritten if we continue)
        final_response = response_text
        
        # Check for tool calls in response
        if use_tools:
            cleaned_text, parsed_tool_calls = parse_tool_calls_from_response(response_text, request.model)
            print(f"🔍 Iteration {iteration}: Found {len(parsed_tool_calls) if parsed_tool_calls else 0} tool call(s)")
            
            # If no tool calls, return the final response
            if not parsed_tool_calls or not use_tools:
                # Clean up tool call markers from response
                final_content = cleaned_text if cleaned_text else final_response
                if not final_content:
                    final_content = final_response
                break
            
            # Execute tool calls
            print(f"🔧 Executing {len(parsed_tool_calls)} tool call(s)...")
            try:
                from ..tool_service import tool_service
                import json
                
                # Execute each tool call
                tool_results = []
                for tool_call in parsed_tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        # Parse arguments
                        arguments_str = tool_call.function.arguments
                        if isinstance(arguments_str, str):
                            arguments = json.loads(arguments_str)
                        else:
                            arguments = arguments_str
                        
                        # Execute tool
                        result = await tool_service.execute_tool(tool_name, arguments)
                        
                        # Format result
                        if isinstance(result, (dict, list)):
                            result_str = json.dumps(result, ensure_ascii=False)
                        else:
                            result_str = str(result)
                        
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": result_str
                        })
                        print(f"✅ Tool '{tool_name}' executed successfully")
                    except Exception as e:
                        error_msg = str(e)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Error: {error_msg}"
                        })
                        print(f"❌ Tool '{tool_name}' execution failed: {error_msg}")
                
                # Add assistant message with tool calls to conversation
                assistant_msg = {
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": [tc.dict() if hasattr(tc, 'dict') else tc for tc in parsed_tool_calls]
                }
                conversation_messages.append(assistant_msg)
                
                # Add tool results to conversation
                tool_results_text = "Tool results:\n"
                for tool_result in tool_results:
                    tool_msg = {
                        "role": "tool",
                        "content": tool_result["content"],
                        "tool_call_id": tool_result["tool_call_id"]
                    }
                    conversation_messages.append(tool_msg)
                    tool_results_text += f"- {tool_result['name']}: {tool_result['content']}\n"
                
                # Create a new user message with tool results to continue the conversation
                continue_msg = {
                    "role": "user",
                    "content": f"Based on the tool results, please provide a final answer:\n{tool_results_text}"
                }
                conversation_messages.append(continue_msg)
                
                # Rebuild prompt with updated conversation for next iteration
                if model_type == "vlm":
                    # For VLM, rebuild from conversation messages
                    prompt, images, _ = extract_content_parts(
                        [ChatMessage(**msg) for msg in conversation_messages],
                        file_storage
                    )
                else:
                    # For LLM, rebuild prompt from conversation
                    tools_to_include = request.tools if use_tools else None
                    if tools_to_include:
                        print(f"🔧 Rebuilding prompt with {len(tools_to_include)} tool(s) for iteration {iteration + 1}")
                    prompt = build_chat_prompt(
                        [ChatMessage(**msg) for msg in conversation_messages],
                        tools_to_include,
                        request.model
                    )
                    images = []
                
                # Continue generation with tool results (next iteration)
                continue
                
            except Exception as e:
                print(f"⚠️  Tool execution error: {e}")
                # Break and return tool_calls for client-side execution
                tool_calls = parsed_tool_calls
                final_content = cleaned_text if cleaned_text else final_response
                finish_reason = "tool_calls"
                break
        else:
            # No tools, just return the response
            final_content = response_text
            break
    
    # If we've exhausted iterations, use the last response we got
    if final_content is None and final_response:
        final_content = final_response
    
    # Apply JSON mode processing if needed
    if use_json_mode and final_content:
        if json_schema:
            is_valid, error_msg = validate_json_schema(final_content, json_schema)
            if not is_valid:
                json_content = extract_json_from_response(final_content)
                if json_content:
                    final_content = json_content
                else:
                    raise HTTPException(status_code=400, detail=f"Response does not match schema: {error_msg}")
            else:
                final_content = extract_json_from_response(final_content) or final_content
        else:
            json_content = extract_json_from_response(final_content)
            if json_content:
                final_content = json_content
    
    # Token estimation
    response_text_for_tokens = final_content or final_response or ""
    prompt_tokens = len(original_prompt.split()) if original_prompt else 0
    completion_tokens = len(response_text_for_tokens.split())
    
    response_message = ChatMessage(role="assistant", content=final_content)
    if tool_calls:
        response_message.tool_calls = tool_calls
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"
    
    # Build conversation_messages for UI (only include tool-related messages if present)
    conversation_for_ui = None
    if tool_calls or any(msg.get("role") == "tool" for msg in conversation_messages):
        conversation_for_ui = []
        # Only include tool-related messages, not all history
        for msg in conversation_messages:
            if msg.get("role") in ["tool"] or (msg.get("role") == "assistant" and msg.get("tool_calls")):
                msg_dict = {
                    "role": msg.get("role"),
                    "content": msg.get("content", ""),
                }
                if msg.get("tool_calls"):
                    msg_dict["tool_calls"] = msg.get("tool_calls")
                if msg.get("tool_call_id"):
                    msg_dict["tool_call_id"] = msg.get("tool_call_id")
                conversation_for_ui.append(msg_dict)
        
        # Add the final assistant response
        final_msg_dict = {
            "role": "assistant",
            "content": final_content or ""
        }
        if tool_calls:
            final_msg_dict["tool_calls"] = [tc.dict() if hasattr(tc, 'dict') else tc for tc in tool_calls]
        conversation_for_ui.append(final_msg_dict)
    
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
                tts_model_name = tts_models[0]
                tts_pipeline = model_manager.tts_pipelines[tts_model_name]
                text_to_speak = final_content or response_text
                
                # Limit text length for TTS (prevent timeout)
                if len(text_to_speak) > 500:
                    text_to_speak = text_to_speak[:500] + "..."
                    print(f"⚠️  TTS: Text truncated to 500 chars for performance")
                
                print(f"🔊 Generating audio with {tts_model_name}...")
                
                # Generate audio
                audio_result = tts_pipeline.generate(text_to_speak)
                
                print(f"   Audio result type: {type(audio_result).__name__}")
                
                # Extract audio data from OpenVINO GenAI result
                audio_array = None
                for attr in ['speeches', 'audios', 'audio', 'waveform', 'data']:
                    if hasattr(audio_result, attr):
                        audio_data = getattr(audio_result, attr)
                        # If it's a list, take the first element
                        audio_array = audio_data[0] if isinstance(audio_data, list) and audio_data else audio_data
                        print(f"   Extracted from .{attr}: type={type(audio_array)}")
                        break
                
                # Fallback to dict access
                if audio_array is None and isinstance(audio_result, dict):
                    for key in ['speeches', 'audio', 'waveform', 'data', 'audios']:
                        if key in audio_result:
                            audio_array = audio_result[key]
                            audio_array = audio_array[0] if isinstance(audio_array, list) and audio_array else audio_array
                            print(f"   Extracted from dict['{key}']: type={type(audio_array)}")
                            break
                
                if audio_array is None:
                    raise ValueError(f"Could not extract audio from TTS result: {type(audio_result)}")
                
                print(f"   Audio shape: {audio_array.shape if hasattr(audio_array, 'shape') else 'N/A'}")
                print(f"   Audio dtype: {audio_array.dtype if hasattr(audio_array, 'dtype') else 'N/A'}")
                
                # Convert to numpy if needed (handle OpenVINO Tensors, torch tensors, etc.)
                if not isinstance(audio_array, np.ndarray):
                    # OpenVINO Tensor - use .data property
                    if hasattr(audio_array, 'data'):
                        audio_array = audio_array.data
                        print(f"   Converted from OpenVINO Tensor via .data")
                    # PyTorch Tensor
                    elif hasattr(audio_array, 'numpy'):
                        audio_array = audio_array.numpy()
                        print(f"   Converted from PyTorch Tensor via .numpy()")
                    elif hasattr(audio_array, 'cpu'):
                        audio_array = audio_array.cpu().numpy()
                        print(f"   Converted from PyTorch Tensor via .cpu().numpy()")
                    else:
                        audio_array = np.array(audio_array)
                        print(f"   Converted via np.array()")
                
                print(f"   After conversion - type: {type(audio_array)}, dtype: {audio_array.dtype if hasattr(audio_array, 'dtype') else 'N/A'}")
                
                # Handle object dtype - force conversion to float32
                if hasattr(audio_array, 'dtype') and (audio_array.dtype == object or audio_array.dtype == np.object_):
                    print(f"   ⚠️  Object dtype detected, converting to float32...")
                    # Try to extract actual numeric data
                    try:
                        # Check if it's a 0-d array containing the actual data
                        if audio_array.ndim == 0:
                            # Extract the scalar value
                            audio_array = audio_array.item()
                            print(f"   Extracted from 0-d array: {type(audio_array)}")
                            
                            # If it's still an object, try to get its data
                            if hasattr(audio_array, 'audios'):
                                audio_array = audio_array.audios[0]
                            elif hasattr(audio_array, 'numpy'):
                                audio_array = audio_array.numpy()
                            
                            # Convert to numpy array
                            if not isinstance(audio_array, np.ndarray):
                                audio_array = np.array(audio_array, dtype=np.float32)
                        elif audio_array.size > 0 and hasattr(audio_array.flat[0], 'numpy'):
                            # If it's an array of arrays or tensors, flatten and convert
                            audio_array = np.concatenate([x.numpy().flatten() for x in audio_array.flat])
                        else:
                            # Last resort: force conversion
                            audio_array = audio_array.astype(np.float32)
                    except Exception as e:
                        print(f"   ⚠️  Object conversion failed: {e}")
                        raise ValueError(f"Cannot convert audio data from object dtype: {type(audio_array)}")
                    print(f"   After object conversion - dtype: {audio_array.dtype}")
                
                # Ensure numeric dtype (soundfile requires float32/64 or int16/32)
                if audio_array.dtype not in [np.float32, np.float64, np.int16, np.int32]:
                    print(f"   Converting {audio_array.dtype} to float32")
                    audio_array = audio_array.astype(np.float32)
                
                # Flatten to 1D if multidimensional
                if audio_array.ndim > 1:
                    print(f"   Flattening {audio_array.ndim}D array to 1D")
                    audio_array = audio_array.flatten()
                
                print(f"   Final audio: shape={audio_array.shape}, dtype={audio_array.dtype}, size={audio_array.size}")
                
                # Convert to base64
                import io
                import soundfile as sf
                audio_buffer = io.BytesIO()
                
                # Validate array
                if audio_array.size == 0:
                    raise ValueError("Audio array is empty after processing")
                
                # Reshape to (samples, channels) for soundfile - it requires 2D for write()
                # Even for mono audio, soundfile expects shape (n_samples, 1)
                audio_array_2d = audio_array.reshape(-1, 1)
                print(f"   Reshaped for soundfile: {audio_array_2d.shape}")
                
                # Write audio (mono channel)
                sf.write(audio_buffer, audio_array_2d, 16000, format='WAV', subtype='PCM_16')
                
                audio_buffer.seek(0)
                audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
                
                # Add audio to response
                response_message.audio = AudioData(
                    id=f"audio_{uuid.uuid4().hex[:16]}",
                    data=audio_base64,
                    transcript=text_to_speak
                )
                print(f"✅ Audio generated: {len(audio_base64)} bytes (base64)")
            except ImportError as e:
                print(f"⚠️  Audio generation skipped: Missing dependency - {e}")
                print(f"    Install: pip install soundfile")
            except Exception as e:
                print(f"⚠️  Audio generation failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Audio output requested but no TTS model loaded")
    
    response_dict = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": response_message.dict(),
            "finish_reason": finish_reason,
            "logprobs": logprobs_data.dict() if logprobs_data else None
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    
    # Add conversation_messages if present
    if conversation_for_ui:
        response_dict["conversation_messages"] = conversation_for_ui
    
    return ChatCompletionResponse(**response_dict)


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