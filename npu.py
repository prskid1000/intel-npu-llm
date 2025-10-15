"""
OpenAI-compatible API Server for OpenVINO GenAI
Supports multiple models running on NPU/CPU/GPU with OpenAI API compatibility
Features:
- Text chat and completion
- Vision/multimodal support (images)
- File uploads and processing
- RAG (Retrieval Augmented Generation)
- Document embeddings and vector search
"""

import asyncio
import json
import time
import uuid
import os
import io
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

import openvino_genai as ov_genai
import openvino as ov
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from PIL import Image


# ============================================================================
# Configuration
# ============================================================================

class ModelConfig(BaseModel):
    """Configuration for a single model"""
    name: str
    path: str
    device: str = "NPU"  # NPU, CPU, or GPU
    type: str = "llm"  # llm, vlm (vision-language), embedding, whisper, or tts


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    models: List[ModelConfig]
    upload_dir: str = "uploads"  # Directory for uploaded files
    vector_store_dir: str = "vector_store"  # Directory for vector embeddings


# Default configuration - can be overridden with config.json
DEFAULT_CONFIG = ServerConfig(
    host="0.0.0.0",
    port=8000,
    models=[
        ModelConfig(
            name="qwen2.5-3b",
            path="models/Qwen/Qwen2.5-3B/1",
            device="NPU"
        )
    ]
)


# ============================================================================
# OpenAI API Request/Response Models
# ============================================================================

class ImageUrl(BaseModel):
    """Image URL or base64 data"""
    url: str
    detail: Optional[str] = "auto"


class MessageContent(BaseModel):
    """Message content part (text or image)"""
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class ChatMessage(BaseModel):
    """Chat message with support for multimodal content"""
    role: str
    content: Union[str, List[MessageContent]]  # String or array of content parts
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "openvino"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class FileObject(BaseModel):
    """Uploaded file object"""
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str  # "assistants", "vision", "fine-tune", etc.


class FileListResponse(BaseModel):
    object: str = "list"
    data: List[FileObject]


class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool


class AudioTranscriptionRequest(BaseModel):
    """Audio transcription request"""
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"  # json, text, srt, verbose_json, vtt
    temperature: Optional[float] = 0.0


class AudioTranscriptionResponse(BaseModel):
    """Audio transcription response"""
    text: str


class AudioSpeechRequest(BaseModel):
    """Text-to-speech request"""
    model: str
    input: str
    voice: Optional[str] = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    response_format: Optional[str] = "mp3"  # mp3, opus, aac, flac, wav, pcm
    speed: Optional[float] = 1.0


# ============================================================================
# File Storage Manager
# ============================================================================

class FileStorageManager:
    """Manages uploaded files and their metadata"""
    
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.files_metadata: Dict[str, FileObject] = {}
        self.metadata_file = self.upload_dir / "files_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load file metadata from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.files_metadata = {
                    k: FileObject(**v) for k, v in data.items()
                }
    
    def _save_metadata(self):
        """Save file metadata to disk"""
        data = {k: v.model_dump() for k, v in self.files_metadata.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def save_file(self, file: UploadFile, purpose: str = "assistants") -> FileObject:
        """Save uploaded file and return metadata"""
        file_id = f"file-{uuid.uuid4().hex}"
        file_path = self.upload_dir / f"{file_id}_{file.filename}"
        
        # Save file to disk
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Create metadata
        file_obj = FileObject(
            id=file_id,
            bytes=len(content),
            created_at=int(time.time()),
            filename=file.filename,
            purpose=purpose
        )
        
        self.files_metadata[file_id] = file_obj
        self._save_metadata()
        
        return file_obj
    
    def get_file(self, file_id: str) -> Optional[FileObject]:
        """Get file metadata"""
        return self.files_metadata.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """Get physical path to file"""
        file_obj = self.get_file(file_id)
        if not file_obj:
            return None
        
        # Find file with matching ID prefix
        for file_path in self.upload_dir.glob(f"{file_id}_*"):
            return file_path
        return None
    
    def list_files(self) -> List[FileObject]:
        """List all uploaded files"""
        return list(self.files_metadata.values())
    
    def delete_file(self, file_id: str) -> bool:
        """Delete file and its metadata"""
        file_path = self.get_file_path(file_id)
        if file_path and file_path.exists():
            file_path.unlink()
        
        if file_id in self.files_metadata:
            del self.files_metadata[file_id]
            self._save_metadata()
            return True
        return False


# ============================================================================
# Document Processor
# ============================================================================

class DocumentProcessor:
    """Extract text from various document formats"""
    
    @staticmethod
    def extract_text(file_path: Path) -> str:
        """Extract text from file based on extension"""
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            # For images, return a placeholder - actual processing done by VLM
            return f"[Image: {file_path.name}]"
        
        elif extension == '.pdf':
            try:
                import PyPDF2
                text = []
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text.append(page.extract_text())
                return '\n'.join(text)
            except ImportError:
                return "[PDF file - PyPDF2 not installed for text extraction]"
        
        elif extension in ['.doc', '.docx']:
            try:
                import docx
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                return "[Word document - python-docx not installed for text extraction]"
        
        else:
            # Try to read as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return f"[Unsupported file type: {extension}]"


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages multiple OpenVINO GenAI pipelines"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.llm_pipelines: Dict[str, ov_genai.LLMPipeline] = {}
        self.vlm_pipelines: Dict[str, ov_genai.VLMPipeline] = {}
        self.whisper_pipelines: Dict[str, ov_genai.WhisperPipeline] = {}
        self.tts_pipelines: Dict[str, ov_genai.Text2SpeechPipeline] = {}
        self.embedding_pipelines: Dict[str, Any] = {}  # For future embedding models
        self.model_configs: Dict[str, ModelConfig] = {}
        
    async def load_models(self):
        """Load all configured models"""
        print("🔄 Loading models...")
        for model_config in self.config.models:
            try:
                print(f"  Loading {model_config.name} ({model_config.type}) on {model_config.device}...")
                
                if model_config.type == "llm":
                    pipeline = ov_genai.LLMPipeline(
                        model_config.path,
                        model_config.device
                    )
                    self.llm_pipelines[model_config.name] = pipeline
                
                elif model_config.type == "vlm":
                    pipeline = ov_genai.VLMPipeline(
                        model_config.path,
                        model_config.device
                    )
                    self.vlm_pipelines[model_config.name] = pipeline
                
                elif model_config.type == "whisper":
                    pipeline = ov_genai.WhisperPipeline(
                        model_config.path,
                        model_config.device
                    )
                    self.whisper_pipelines[model_config.name] = pipeline
                
                elif model_config.type == "tts":
                    pipeline = ov_genai.Text2SpeechPipeline(
                        model_config.path,
                        model_config.device
                    )
                    self.tts_pipelines[model_config.name] = pipeline
                
                elif model_config.type == "embedding":
                    # Placeholder for embedding models
                    print(f"  ⚠️  Embedding models not fully implemented yet")
                    continue
                
                else:
                    print(f"  ⚠️  Unknown model type: {model_config.type}")
                    continue
                
                self.model_configs[model_config.name] = model_config
                print(f"  ✅ {model_config.name} loaded successfully")
                
            except Exception as e:
                print(f"  ❌ Failed to load {model_config.name}: {e}")
                
        total_models = (len(self.llm_pipelines) + len(self.vlm_pipelines) + 
                       len(self.whisper_pipelines) + len(self.tts_pipelines))
        if total_models == 0:
            raise RuntimeError("No models loaded successfully!")
            
        print(f"✅ Loaded {total_models} model(s)")
        
    def get_pipeline(self, model_name: str) -> Any:
        """Get a pipeline by model name"""
        if model_name in self.llm_pipelines:
            return self.llm_pipelines[model_name]
        elif model_name in self.vlm_pipelines:
            return self.vlm_pipelines[model_name]
        elif model_name in self.whisper_pipelines:
            return self.whisper_pipelines[model_name]
        elif model_name in self.tts_pipelines:
            return self.tts_pipelines[model_name]
        else:
            all_models = (list(self.llm_pipelines.keys()) + list(self.vlm_pipelines.keys()) +
                         list(self.whisper_pipelines.keys()) + list(self.tts_pipelines.keys()))
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {all_models}"
            )
    
    def get_model_type(self, model_name: str) -> str:
        """Get the type of a model"""
        if model_name in self.model_configs:
            return self.model_configs[model_name].type
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return (list(self.llm_pipelines.keys()) + list(self.vlm_pipelines.keys()) +
                list(self.whisper_pipelines.keys()) + list(self.tts_pipelines.keys()))
    
    async def cleanup(self):
        """Cleanup all loaded models"""
        print("🔄 Cleaning up models...")
        self.llm_pipelines.clear()
        self.vlm_pipelines.clear()
        self.whisper_pipelines.clear()
        self.tts_pipelines.clear()
        self.embedding_pipelines.clear()
        self.model_configs.clear()
        print("✅ Cleanup complete")


# ============================================================================
# Application Lifecycle
# ============================================================================

# Global managers
model_manager: Optional[ModelManager] = None
file_storage: Optional[FileStorageManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global model_manager, file_storage
    
    # Startup: Load configuration and models
    try:
        # Try to load config.json, otherwise use defaults
        try:
            with open("config.json", "r") as f:
                config_data = json.load(f)
                config = ServerConfig(**config_data)
                print("📋 Loaded configuration from config.json")
        except FileNotFoundError:
            config = DEFAULT_CONFIG
            print("📋 Using default configuration")
        
        # Initialize file storage
        file_storage = FileStorageManager(config.upload_dir)
        print(f"📁 File storage initialized at: {config.upload_dir}")
        
        # Initialize model manager    
        model_manager = ModelManager(config)
        await model_manager.load_models()
        
        print(f"🚀 Server ready at http://{config.host}:{config.port}")
        print(f"📚 Available models: {', '.join(model_manager.list_models())}")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    if model_manager:
        await model_manager.cleanup()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="OpenVINO GenAI Server",
    description="OpenAI-compatible API server for OpenVINO GenAI models",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenVINO GenAI Server",
        "version": "1.0.0",
        "models": model_manager.list_models() if model_manager else []
    }


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List available models (OpenAI-compatible)"""
    models = [
        ModelInfo(
            id=model_name,
            created=int(time.time())
        )
        for model_name in model_manager.list_models()
    ]
    return ModelListResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Create chat completion (OpenAI-compatible) with multimodal support"""
    
    # Get the pipeline
    pipeline = model_manager.get_pipeline(request.model)
    model_type = model_manager.get_model_type(request.model)
    
    # Extract content (text, images, files)
    prompt, images, file_ids = extract_content_parts(request.messages)
    
    # Configure generation
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = request.max_tokens
    config.temperature = request.temperature
    config.top_p = request.top_p
    config.do_sample = request.temperature > 0
    
    # Process any file attachments for RAG context
    rag_context = ""
    if file_ids:
        for file_id in file_ids:
            file_path = file_storage.get_file_path(file_id)
            if file_path:
                try:
                    extracted_text = DocumentProcessor.extract_text(file_path)
                    if extracted_text and not extracted_text.startswith("["):
                        rag_context += f"\n\n---Document Content ({file_path.name})---\n{extracted_text}\n"
                except Exception as e:
                    print(f"Error extracting text from {file_id}: {e}")
    
    # Prepend RAG context if available
    if rag_context:
        prompt = f"{rag_context}\n\n{prompt}"
    
    # Generate response based on model type
    if request.stream:
        if model_type == "vlm" and images:
            # VLM streaming not fully implemented, fall back to non-streaming
            return await chat_completions_non_streaming(pipeline, model_type, prompt, images, request, config)
        else:
            return StreamingResponse(
                stream_chat_completion(pipeline, prompt, request, config),
                media_type="text/event-stream"
            )
    else:
        return await chat_completions_non_streaming(pipeline, model_type, prompt, images, request, config)


async def chat_completions_non_streaming(
    pipeline: Union[ov_genai.LLMPipeline, ov_genai.VLMPipeline],
    model_type: str,
    prompt: str,
    images: List[ov.Tensor],
    request: ChatCompletionRequest,
    config: ov_genai.GenerationConfig
) -> ChatCompletionResponse:
    """Non-streaming chat completion handler"""
    
    start_time = time.time()
    
    # Generate based on model type
    if model_type == "vlm" and images:
        # Vision-language model with images
        # Use the first image (can be extended for multiple images)
        image_tensor = images[0] if images else None
        if image_tensor is not None:
            result = pipeline.generate(prompt, image=image_tensor, max_new_tokens=config.max_new_tokens)
            response_text = result if isinstance(result, str) else result.texts[0]
        else:
            response_text = pipeline.generate(prompt, config)
    else:
        # Text-only LLM
        response_text = pipeline.generate(prompt, config)
    
    generation_time = time.time() - start_time
    
    # Estimate tokens (rough approximation)
    prompt_tokens = len(prompt.split())
    completion_tokens = len(response_text.split())
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Create text completion (OpenAI-compatible)"""
    
    # Get the pipeline
    pipeline = model_manager.get_pipeline(request.model)
    
    # Handle prompt (can be string or list)
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    
    # Configure generation
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = request.max_tokens
    config.temperature = request.temperature
    config.top_p = request.top_p
    config.do_sample = request.temperature > 0
    
    # Generate response
    if request.stream:
        return StreamingResponse(
            stream_completion(pipeline, prompt, request, config),
            media_type="text/event-stream"
        )
    else:
        response_text = pipeline.generate(prompt, config)
        
        # Estimate tokens
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=response_text,
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    llm_count = len(model_manager.llm_pipelines) if model_manager else 0
    vlm_count = len(model_manager.vlm_pipelines) if model_manager else 0
    whisper_count = len(model_manager.whisper_pipelines) if model_manager else 0
    tts_count = len(model_manager.tts_pipelines) if model_manager else 0
    
    return {
        "status": "healthy",
        "models_loaded": llm_count + vlm_count + whisper_count + tts_count,
        "llm_models": llm_count,
        "vlm_models": vlm_count,
        "whisper_models": whisper_count,
        "tts_models": tts_count,
        "files_stored": len(file_storage.files_metadata) if file_storage else 0
    }


# ============================================================================
# File Management Endpoints
# ============================================================================

@app.post("/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("assistants")
) -> FileObject:
    """Upload a file (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    file_obj = await file_storage.save_file(file, purpose)
    return file_obj


@app.get("/v1/files")
async def list_files() -> FileListResponse:
    """List all uploaded files (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    files = file_storage.list_files()
    return FileListResponse(data=files)


@app.get("/v1/files/{file_id}")
async def get_file(file_id: str) -> FileObject:
    """Get file metadata (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    file_obj = file_storage.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    
    return file_obj


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str) -> FileDeleteResponse:
    """Delete a file (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    deleted = file_storage.delete_file(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    
    return FileDeleteResponse(id=file_id, deleted=True)


@app.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str):
    """Get file content (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    file_path = file_storage.get_file_path(file_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(file_path)


# ============================================================================
# Audio Endpoints
# ============================================================================

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    """Transcribe audio to text using Whisper (OpenAI-compatible)"""
    
    # Get the Whisper pipeline
    pipeline = model_manager.get_pipeline(model)
    model_type = model_manager.get_model_type(model)
    
    if model_type != "whisper":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not a Whisper model. Use a model with type 'whisper'."
        )
    
    # Read audio file
    audio_bytes = await file.read()
    
    # Save to temporary file for processing
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_audio_path = temp_file.name
    
    try:
        # Load audio with librosa
        import librosa
        raw_speech, samplerate = librosa.load(temp_audio_path, sr=16000)
        raw_speech_list = raw_speech.tolist()
        
        # Transcribe
        transcription = pipeline.generate(raw_speech_list)
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        # Return based on response format
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
            # For srt, vtt formats - simplified version
            return AudioTranscriptionResponse(text=transcription)
            
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """Generate speech from text using TTS (OpenAI-compatible)"""
    
    # Get the TTS pipeline
    pipeline = model_manager.get_pipeline(request.model)
    model_type = model_manager.get_model_type(request.model)
    
    if model_type != "tts":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not a TTS model. Use a model with type 'tts'."
        )
    
    try:
        # Generate speech
        result = pipeline.generate(request.input)
        speech = result.speeches[0]
        audio_data = speech.data[0]
        
        # Get sample rate (default 16000 for most TTS models)
        sample_rate = 16000
        
        # Convert to requested format
        import tempfile
        import soundfile as sf
        
        # Create temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_wav_path = temp_file.name
            sf.write(temp_wav_path, audio_data, samplerate=sample_rate)
        
        # Convert to requested format if needed
        output_path = temp_wav_path
        
        if request.response_format != "wav":
            # Use pydub for format conversion if available
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(temp_wav_path)
                
                output_path = temp_wav_path.replace(".wav", f".{request.response_format}")
                
                if request.response_format == "mp3":
                    audio.export(output_path, format="mp3")
                elif request.response_format == "opus":
                    audio.export(output_path, format="opus")
                elif request.response_format == "aac":
                    audio.export(output_path, format="aac")
                elif request.response_format == "flac":
                    audio.export(output_path, format="flac")
                elif request.response_format == "pcm":
                    # Raw PCM data
                    with open(output_path, 'wb') as f:
                        f.write(audio_data.tobytes())
                
                # Clean up temp wav
                os.unlink(temp_wav_path)
                
            except ImportError:
                # If pydub not available, return wav
                print("⚠️  pydub not installed, returning wav format")
        
        # Return audio file
        from fastapi.responses import FileResponse
        
        media_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        media_type = media_types.get(request.response_format, "audio/wav")
        
        # Create response with cleanup
        response = FileResponse(
            output_path,
            media_type=media_type,
            filename=f"speech.{request.response_format}"
        )
        
        # Schedule cleanup after response is sent
        import asyncio
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


# ============================================================================
# Helper Functions
# ============================================================================

def load_image_from_url(url: str) -> ov.Tensor:
    """Load image from URL or base64 data URI"""
    import base64
    import requests
    from io import BytesIO
    
    if url.startswith('data:image'):
        # Base64 encoded image
        header, encoded = url.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
    elif url.startswith('http://') or url.startswith('https://'):
        # Remote URL
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    elif url.startswith('file://'):
        # Local file
        file_path = url[7:]  # Remove 'file://'
        image = Image.open(file_path)
    elif url.startswith('file-'):
        # Uploaded file ID
        if not file_storage:
            raise HTTPException(status_code=500, detail="File storage not initialized")
        file_path = file_storage.get_file_path(url)
        if not file_path:
            raise HTTPException(status_code=404, detail=f"File '{url}' not found")
        image = Image.open(file_path)
    else:
        # Assume local file path
        image = Image.open(url)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and then to OpenVINO tensor
    image_array = np.array(image)
    return ov.Tensor(image_array)


def extract_content_parts(messages: List[ChatMessage]) -> tuple[str, List[ov.Tensor], List[str]]:
    """Extract text, images, and file references from messages
    
    Returns:
        tuple: (text_prompt, image_tensors, file_ids)
    """
    text_parts = []
    images = []
    file_ids = []
    
    for msg in messages:
        role_prefix = ""
        if msg.role == "system":
            role_prefix = "System: "
        elif msg.role == "user":
            role_prefix = "User: "
        elif msg.role == "assistant":
            role_prefix = "Assistant: "
        
        # Handle string content
        if isinstance(msg.content, str):
            text_parts.append(f"{role_prefix}{msg.content}\n\n")
        
        # Handle array of content parts
        elif isinstance(msg.content, list):
            msg_text = []
            for part in msg.content:
                if part.type == "text" and part.text:
                    msg_text.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    try:
                        img_tensor = load_image_from_url(part.image_url.url)
                        images.append(img_tensor)
                        msg_text.append("[Image attached]")
                        
                        # Track if it's a file ID
                        if part.image_url.url.startswith('file-'):
                            file_ids.append(part.image_url.url)
                    except Exception as e:
                        msg_text.append(f"[Image load error: {e}]")
            
            if msg_text:
                text_parts.append(f"{role_prefix}{' '.join(msg_text)}\n\n")
    
    # Combine text
    text_parts.append("Assistant: ")
    full_prompt = "".join(text_parts)
    
    return full_prompt, images, file_ids


def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string (text-only)"""
    formatted = ""
    for msg in messages:
        content_text = ""
        
        if isinstance(msg.content, str):
            content_text = msg.content
        elif isinstance(msg.content, list):
            # Extract only text parts
            text_parts = [part.text for part in msg.content if part.type == "text" and part.text]
            content_text = " ".join(text_parts)
        
        if msg.role == "system":
            formatted += f"System: {content_text}\n\n"
        elif msg.role == "user":
            formatted += f"User: {content_text}\n\n"
        elif msg.role == "assistant":
            formatted += f"Assistant: {content_text}\n\n"
    
    formatted += "Assistant: "
    return formatted


async def stream_chat_completion(
    pipeline: ov_genai.LLMPipeline,
    prompt: str,
    request: ChatCompletionRequest,
    config: ov_genai.GenerationConfig
):
    """Stream chat completion responses"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    # Send initial chunk
    initial_chunk = ChatCompletionStreamResponse(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={"role": "assistant", "content": ""},
                finish_reason=None
            )
        ]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"
    
    # Create a custom streamer class
    class TokenStreamer:
        def __init__(self):
            self.tokens = []
        
        def __call__(self, token: str) -> int:
            self.tokens.append(token)
            return 0  # Continue generation
    
    # Generate with custom streamer
    streamer = TokenStreamer()
    result = pipeline.generate(prompt, config, streamer=streamer)
    
    # Stream the collected tokens
    for token in streamer.tokens:
        chunk = ChatCompletionStreamResponse(
            id=chunk_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": token},
                    finish_reason=None
                )
            ]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0)  # Allow other tasks to run
    
    # Send final chunk
    final_chunk = ChatCompletionStreamResponse(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def stream_completion(
    pipeline: ov_genai.LLMPipeline,
    prompt: str,
    request: CompletionRequest,
    config: ov_genai.GenerationConfig
):
    """Stream text completion responses"""
    chunk_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    # Create a custom streamer class
    class TokenStreamer:
        def __init__(self):
            self.tokens = []
        
        def __call__(self, token: str) -> int:
            self.tokens.append(token)
            return 0  # Continue generation
    
    # Generate with custom streamer
    streamer = TokenStreamer()
    result = pipeline.generate(prompt, config, streamer=streamer)
    
    # Stream the collected tokens
    for token in streamer.tokens:
        chunk = {
            "id": chunk_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": token,
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)
    
    # Send final chunk
    final_chunk = {
        "id": chunk_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the server"""
    # Load config to get host/port
    try:
        with open("config.json", "r") as f:
            config_data = json.load(f)
            config = ServerConfig(**config_data)
    except FileNotFoundError:
        config = DEFAULT_CONFIG
    
    uvicorn.run(
        "npu:app",
        host=config.host,
        port=config.port,
        reload=False
    )


if __name__ == "__main__":
    main()

