"""
Audio API Routes
POST /v1/audio/transcriptions - Speech-to-text (Whisper)
POST /v1/audio/speech - Text-to-speech
"""

import os
import time
import tempfile
import asyncio
import numpy as np
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
        
        print(f"   Audio result type: {type(result).__name__}")
        
        # Extract audio data from OpenVINO GenAI result
        audio_array = None
        for attr in ['speeches', 'audios', 'audio', 'waveform', 'data']:
            if hasattr(result, attr):
                audio_data = getattr(result, attr)
                # If it's a list, take the first element
                audio_array = audio_data[0] if isinstance(audio_data, list) and audio_data else audio_data
                print(f"   Extracted from .{attr}: type={type(audio_array)}")
                break
        
        # Fallback to dict access
        if audio_array is None and isinstance(result, dict):
            for key in ['speeches', 'audio', 'waveform', 'data', 'audios']:
                if key in result:
                    audio_array = result[key]
                    audio_array = audio_array[0] if isinstance(audio_array, list) and audio_array else audio_array
                    print(f"   Extracted from dict['{key}']: type={type(audio_array)}")
                    break
        
        if audio_array is None:
            raise ValueError(f"Could not extract audio from TTS result: {type(result)}")
        
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
        
        # Validate array
        if audio_array.size == 0:
            raise ValueError("Audio array is empty after processing")
        
        sample_rate = 16000
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_wav_path = temp_file.name
            # Reshape to (samples, channels) for soundfile - it requires 2D for write()
            audio_array_2d = audio_array.reshape(-1, 1)
            print(f"   Reshaped for soundfile: {audio_array_2d.shape}")
            sf.write(temp_wav_path, audio_array_2d, samplerate=sample_rate, format='WAV', subtype='PCM_16')
        
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

