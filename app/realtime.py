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
from .utils import build_chat_prompt

if TYPE_CHECKING:
    from .managers import ModelManager

import openvino_genai as ov_genai


class RealtimeSession:
    """Manages a realtime voice/text conversation session"""
    
    def __init__(self, websocket: WebSocket, model_name: str, model_manager):
        self.websocket = websocket
        self.model_name = model_name
        self.model_manager = model_manager
        self.session_id = f"sess_{uuid.uuid4().hex[:16]}"
        self.conversation_history: List[Dict[str, Any]] = []
        self.audio_buffer: List[bytes] = []
        self.is_speaking = False
        
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
    
    async def generate_response(self, user_input: str):
        """Generate and stream text/audio response"""
        try:
            pipeline = self.model_manager.get_pipeline(self.model_name)
            
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 512
            config.temperature = 0.7
            config.do_sample = True
            
            prompt = build_chat_prompt(
                [ChatMessage(role=msg["role"], content=msg["content"]) 
                 for msg in self.conversation_history],
                tools=None
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
                for token in pipeline.generate(prompt, config, streamer=True):
                    response_text += token
                    await self.send_event("response.text.delta", {
                        "response_id": response_id,
                        "delta": token,
                        "text": response_text
                    })
                    await asyncio.sleep(0)
                    
            except TypeError:
                response_text = pipeline.generate(prompt, config)
                await self.send_event("response.text.delta", {
                    "response_id": response_id,
                    "delta": response_text,
                    "text": response_text
                })
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            await self.send_event("response.done", {
                "response": {
                    "id": response_id,
                    "status": "completed",
                    "output": [{"type": "text", "text": response_text}]
                }
            })
            
            tts_models = list(self.model_manager.tts_pipelines.keys())
            if tts_models:
                await self.generate_audio_response(response_text, response_id)
                
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


async def realtime_endpoint(websocket: WebSocket, model: str, model_manager):
    """OpenAI-compatible Realtime API (WebSocket)"""
    await websocket.accept()
    
    session = RealtimeSession(websocket, model, model_manager)
    
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
                    
                    if event_type == "conversation.item.create":
                        item = event.get("item", {})
                        content = item.get("content", [])
                        
                        for part in content:
                            if part.get("type") == "input_text":
                                text = part.get("text", "")
                                session.conversation_history.append({
                                    "role": "user",
                                    "content": text
                                })
                                await session.generate_response(text)
                    
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
        try:
            await websocket.close()
        except:
            pass

