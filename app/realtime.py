"""
WebSocket Realtime API
Full-duplex voice+text conversation
"""

import asyncio
import json
import uuid
import os
import base64
import tempfile
from typing import List, Dict, Any, TYPE_CHECKING
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from .models import ChatMessage
from .utils import build_chat_prompt, parse_tool_calls_from_response
from .session_manager import Session, SessionManager

if TYPE_CHECKING:
    from .managers import ModelManager

import openvino_genai as ov_genai


class RealtimeSession:
    """
    Wrapper for realtime session with model manager and WebSocket-specific methods
    Extends the Session class with generation capabilities
    """
    
    def __init__(self, session: Session, model_manager):
        self.session = session
        self.websocket = session.websocket
        self.model_name = session.model_name
        self.model_manager = model_manager
        self.session_id = session.session_id
        self.conversation_history = session.conversation_history
        self.audio_buffer = session.audio_buffer
        self.is_speaking = session.is_speaking
        self.tools = session.tools
        
    async def send_event(self, event_type: str, data: Dict[str, Any]):
        """Send event to client"""
        event = {
            "type": event_type,
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            **data
        }
        await self.websocket.send_json(event)
    
    async def send_error(self, message: str, code: str = "server_error"):
        """Send error event"""
        await self.send_event("error", {
            "error": {
                "type": code,
                "message": message
            }
        })
    
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk to buffer"""
        self.audio_buffer.append(audio_data)
    
    async def process_audio_input(self):
        """Process accumulated audio input and generate response"""
        if not self.audio_buffer:
            return
        
        try:
            audio_bytes = b''.join(self.audio_buffer)
            self.audio_buffer.clear()
            
            whisper_models = list(self.model_manager.whisper_pipelines.keys())
            if whisper_models:
                import librosa
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_path = temp_file.name
                
                try:
                    raw_speech, samplerate = librosa.load(temp_path, sr=16000)
                    whisper_pipeline = self.model_manager.whisper_pipelines[whisper_models[0]]
                    transcription = whisper_pipeline.generate(raw_speech.tolist())
                    
                    os.unlink(temp_path)
                    
                    await self.send_event("conversation.item.created", {
                        "item": {
                            "id": f"msg_{uuid.uuid4().hex[:12]}",
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": transcription}]
                        }
                    })
                    
                    self.conversation_history.append({
                        "role": "user",
                        "content": transcription
                    })
                    
                    await self.generate_response(transcription)
                    
                except Exception as e:
                    await self.send_error(f"Audio processing failed: {str(e)}")
            else:
                await self.send_error("No Whisper model available")
                
        except Exception as e:
            await self.send_error(f"Failed to process audio: {str(e)}")
    
    async def generate_response(self, user_input: str, images: list = None):
        """Generate and stream text/audio response with optional image support"""
        try:
            pipeline = self.model_manager.get_pipeline(self.model_name)
            model_type = self.model_manager.get_model_type(self.model_name)
            
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 512
            config.temperature = 0.7
            config.do_sample = True
            
            prompt = build_chat_prompt(
                [ChatMessage(role=msg["role"], content=msg["content"]) 
                 for msg in self.conversation_history],
                tools=self.tools if self.tools else None
            )
            
            response_id = f"resp_{uuid.uuid4().hex[:12]}"
            await self.send_event("response.created", {
                "response": {
                    "id": response_id,
                    "status": "in_progress"
                }
            })
            
            response_text = ""
            
            try:
                # VLM models require images parameter
                if model_type == "vlm":
                    import openvino as ov
                    from PIL import Image
                    import base64
                    import io
                    
                    # Process images if provided
                    image_tensors = []
                    if images:
                        for img_data in images:
                            if img_data.startswith('data:image'):
                                # Base64 encoded image
                                img_data = img_data.split(',')[1]
                                img_bytes = base64.b64decode(img_data)
                                img = Image.open(io.BytesIO(img_bytes))
                            else:
                                # Assume it's a file path or URL
                                img = Image.open(img_data)
                            
                            # Resize and convert to tensor
                            img = img.resize((224, 224))
                            import numpy as np
                            img_array = np.array(img)
                            if len(img_array.shape) == 2:
                                img_array = np.stack([img_array] * 3, axis=2)
                            img_tensor = ov.Tensor(img_array.transpose(2, 0, 1).astype(np.uint8))
                            image_tensors.append(img_tensor)
                    else:
                        # Create empty image tensor for text-only
                        empty_image = ov.Tensor(ov.Type.u8, [3, 224, 224])
                        image_tensors = [empty_image]
                    
                    response_text = pipeline.generate(prompt, image_tensors, config)
                else:
                    # Regular LLM models
                    for token in pipeline.generate(prompt, config, streamer=True):
                        response_text += token
                        await self.send_event("response.text.delta", {
                            "response_id": response_id,
                            "delta": token,
                            "text": response_text
                        })
                        await asyncio.sleep(0)
                    
            except TypeError:
                # Fallback for models without streaming
                if model_type == "vlm":
                    import openvino as ov
                    empty_image = ov.Tensor(ov.Type.u8, [3, 224, 224])
                    response_text = pipeline.generate(prompt, [empty_image], config)
                else:
                    response_text = pipeline.generate(prompt, config)
                    
                await self.send_event("response.text.delta", {
                    "response_id": response_id,
                    "delta": response_text,
                    "text": response_text
                })
            
            # Check for function calls if tools are available
            tool_calls = None
            final_text = response_text
            
            if self.tools:
                cleaned_text, parsed_tool_calls = parse_tool_calls_from_response(response_text)
                if parsed_tool_calls:
                    tool_calls = parsed_tool_calls
                    final_text = cleaned_text
                    
                    # Send function call events
                    for tool_call in tool_calls:
                        await self.send_event("response.function_call_arguments.done", {
                            "response_id": response_id,
                            "call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        })
            
            self.conversation_history.append({
                "role": "assistant",
                "content": final_text,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ] if tool_calls else None
            })
            
            output = [{"type": "text", "text": final_text}]
            if tool_calls:
                output.append({
                    "type": "function_calls",
                    "function_calls": [
                        {
                            "call_id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        } for tc in tool_calls
                    ]
                })
            
            await self.send_event("response.done", {
                "response": {
                    "id": response_id,
                    "status": "completed",
                    "output": output
                }
            })
            
            # Only generate audio if there's text and no function calls
            if final_text and not tool_calls:
                tts_models = list(self.model_manager.tts_pipelines.keys())
                if tts_models:
                    await self.generate_audio_response(final_text, response_id)
                
        except Exception as e:
            await self.send_error(f"Response generation failed: {str(e)}")
    
    async def generate_audio_response(self, text: str, response_id: str):
        """Generate and stream audio for text response"""
        try:
            tts_models = list(self.model_manager.tts_pipelines.keys())
            if not tts_models:
                return
            
            tts_pipeline = self.model_manager.tts_pipelines[tts_models[0]]
            
            result = tts_pipeline.generate(text)
            speech = result.speeches[0]
            audio_data = speech.data[0]
            
            sample_rate = 16000
            chunk_size = 4096
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                audio_bytes = chunk.astype(np.int16).tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                await self.send_event("response.audio.delta", {
                    "response_id": response_id,
                    "delta": audio_b64,
                    "sample_rate": sample_rate
                })
                
                await asyncio.sleep(0.01)
            
            await self.send_event("response.audio.done", {
                "response_id": response_id
            })
            
        except Exception as e:
            await self.send_error(f"Audio generation failed: {str(e)}")


async def realtime_endpoint(websocket: WebSocket, model: str, model_manager, session_manager: SessionManager):
    """OpenAI-compatible Realtime API (WebSocket)"""
    await websocket.accept()
    
    # Create session using SessionManager
    base_session = session_manager.create_session(model, websocket)
    session = RealtimeSession(base_session, model_manager)
    
    try:
        await session.send_event("session.created", {
            "session": {
                "id": session.session_id,
                "model": model,
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        })
        
        while True:
            try:
                data = await websocket.receive()
                
                if "text" in data:
                    event = json.loads(data["text"])
                    event_type = event.get("type")
                    
                    if event_type == "session.update":
                        # Update session configuration (e.g., set tools)
                        session_config = event.get("session", {})
                        if "tools" in session_config:
                            session.tools = session_config["tools"]
                            await session.send_event("session.updated", {
                                "session": {
                                    "id": session.session_id,
                                    "tools": session.tools
                                }
                            })
                    
                    elif event_type == "conversation.item.create":
                        item = event.get("item", {})
                        item_type = item.get("type", "message")
                        
                        # Handle function call results
                        if item_type == "function_call_output":
                            call_id = item.get("call_id")
                            output = item.get("output", "")
                            
                            # Add function result to conversation
                            session.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": output
                            })
                            
                            # Optionally continue the conversation automatically
                            # await session.generate_response("")
                            
                            await session.send_event("conversation.item.created", {
                                "item": {
                                    "id": f"msg_{uuid.uuid4().hex[:12]}",
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": output
                                }
                            })
                        else:
                            # Regular message
                            content = item.get("content", [])
                            
                            # Support multimodal input (text + images)
                            text_parts = []
                            image_parts = []
                            
                            for part in content:
                                if part.get("type") == "input_text":
                                    text_parts.append(part.get("text", ""))
                                elif part.get("type") == "input_image":
                                    # Extract image data (base64 or URL)
                                    image_url = part.get("image_url", {})
                                    if isinstance(image_url, dict):
                                        url = image_url.get("url", "")
                                    else:
                                        url = image_url
                                    if url:
                                        image_parts.append(url)
                            
                            if text_parts:
                                text = " ".join(text_parts)
                                session.conversation_history.append({
                                    "role": "user",
                                    "content": text
                                })
                                await session.generate_response(text, images=image_parts)
                    
                    elif event_type == "input_audio_buffer.append":
                        audio_b64 = event.get("audio", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            session.add_audio_chunk(audio_bytes)
                    
                    elif event_type == "input_audio_buffer.commit":
                        await session.process_audio_input()
                    
                    elif event_type == "input_audio_buffer.clear":
                        session.audio_buffer.clear()
                    
                    elif event_type == "response.cancel":
                        session.is_speaking = False
                
                elif "bytes" in data:
                    audio_bytes = data["bytes"]
                    session.add_audio_chunk(audio_bytes)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                await session.send_error(f"Event processing failed: {str(e)}")
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await session.send_error(f"Session error: {str(e)}")
        except:
            pass
    finally:
        print(f"Session {session.session_id} ended")
        # Keep session in memory for potential reconnection (will auto-expire after timeout)
        # To delete immediately, uncomment: session_manager.delete_session(session.session_id)
        try:
            await websocket.close()
        except:
            pass

