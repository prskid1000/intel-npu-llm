"""
Audio API Routes
POST /v1/audio/transcriptions - Speech-to-text (Whisper)
POST /v1/audio/speech - Text-to-speech
"""

import os
import time
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse

from ..models import AudioTranscriptionResponse, AudioSpeechRequest

if TYPE_CHECKING:
    from ..managers import ModelManager

router = APIRouter()
model_manager: 'ModelManager' = None


def set_model_manager(manager):
    global model_manager
    model_manager = manager


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    """Transcribe audio to text using Whisper (OpenAI-compatible)"""
    
    print(f"🎤 STT: model={model}, file={file.filename}")
    
    pipeline = model_manager.get_pipeline(model)
    model_type = model_manager.get_model_type(model)
    
    if model_type != "whisper":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not a Whisper model"
        )
    
    audio_bytes = await file.read()
    
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_audio_path = temp_file.name
    
    try:
        import librosa
        raw_speech, samplerate = librosa.load(temp_audio_path, sr=16000)
        result = pipeline.generate(raw_speech.tolist())
        
        # Extract text from WhisperDecodedResults object
        if hasattr(result, 'texts'):
            transcription = result.texts[0] if result.texts else ""
        elif hasattr(result, '__str__'):
            transcription = str(result)
        else:
            transcription = result
        
        os.unlink(temp_audio_path)
        
        if response_format == "text":
            return transcription
        elif response_format == "json":
            return AudioTranscriptionResponse(text=transcription)
        elif response_format == "verbose_json":
            return {
                "text": transcription,
                "task": "transcribe",
                "language": language or "auto",
                "duration": len(raw_speech) / samplerate
            }
        else:
            return AudioTranscriptionResponse(text=transcription)
            
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """Generate speech from text using TTS (OpenAI-compatible)"""
    
    print(f"🔊 TTS: model={request.model}, text_len={len(request.input)}")
    
    pipeline = model_manager.get_pipeline(request.model)
    model_type = model_manager.get_model_type(request.model)
    
    if model_type != "tts":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not a TTS model"
        )
    
    try:
        result = pipeline.generate(request.input)
        speech = result.speeches[0]
        audio_data = speech.data[0]
        sample_rate = 16000
        
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_wav_path = temp_file.name
            sf.write(temp_wav_path, audio_data, samplerate=sample_rate)
        
        output_path = temp_wav_path
        
        if request.response_format != "wav":
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(temp_wav_path)
                output_path = temp_wav_path.replace(".wav", f".{request.response_format}")
                
                if request.response_format == "mp3":
                    audio.export(output_path, format="mp3")
                elif request.response_format in ["opus", "aac", "flac"]:
                    audio.export(output_path, format=request.response_format)
                elif request.response_format == "pcm":
                    with open(output_path, 'wb') as f:
                        f.write(audio_data.tobytes())
                
                os.unlink(temp_wav_path)
            except ImportError:
                print("⚠️  pydub not installed, returning wav format")
        
        media_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        media_type = media_types.get(request.response_format, "audio/wav")
        response = FileResponse(
            output_path,
            media_type=media_type,
            filename=f"speech.{request.response_format}"
        )
        
        asyncio.create_task(cleanup_temp_file(output_path))
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


async def cleanup_temp_file(file_path: str, delay: float = 1.0):
    """Cleanup temporary file after a delay"""
    await asyncio.sleep(delay)
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except:
            pass

