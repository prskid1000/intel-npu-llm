"""
OpenAI-compatible API Server for OpenVINO GenAI
Supports multiple models running on NPU/CPU/GPU with OpenAI API compatibility
"""

import asyncio
import json
import time
import uuid
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

import openvino_genai as ov_genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn


# ============================================================================
# Configuration
# ============================================================================

class ModelConfig(BaseModel):
    """Configuration for a single model"""
    name: str
    path: str
    device: str = "NPU"  # NPU, CPU, or GPU


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    models: List[ModelConfig]


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

class ChatMessage(BaseModel):
    role: str
    content: str


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


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages multiple OpenVINO GenAI pipelines"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.pipelines: Dict[str, ov_genai.LLMPipeline] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        
    async def load_models(self):
        """Load all configured models"""
        print("🔄 Loading models...")
        for model_config in self.config.models:
            try:
                print(f"  Loading {model_config.name} on {model_config.device}...")
                pipeline = ov_genai.LLMPipeline(
                    model_config.path,
                    model_config.device
                )
                self.pipelines[model_config.name] = pipeline
                self.model_configs[model_config.name] = model_config
                print(f"  ✅ {model_config.name} loaded successfully")
            except Exception as e:
                print(f"  ❌ Failed to load {model_config.name}: {e}")
                
        if not self.pipelines:
            raise RuntimeError("No models loaded successfully!")
            
        print(f"✅ Loaded {len(self.pipelines)} model(s)")
        
    def get_pipeline(self, model_name: str) -> ov_genai.LLMPipeline:
        """Get a pipeline by model name"""
        if model_name not in self.pipelines:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {list(self.pipelines.keys())}"
            )
        return self.pipelines[model_name]
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.pipelines.keys())
    
    async def cleanup(self):
        """Cleanup all loaded models"""
        print("🔄 Cleaning up models...")
        self.pipelines.clear()
        self.model_configs.clear()
        print("✅ Cleanup complete")


# ============================================================================
# Application Lifecycle
# ============================================================================

# Global model manager
model_manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global model_manager
    
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
    """Create chat completion (OpenAI-compatible)"""
    
    # Get the pipeline
    pipeline = model_manager.get_pipeline(request.model)
    
    # Convert messages to prompt
    prompt = format_chat_prompt(request.messages)
    
    # Configure generation
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = request.max_tokens
    config.temperature = request.temperature
    config.top_p = request.top_p
    config.do_sample = request.temperature > 0
    
    # Generate response
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(pipeline, prompt, request, config),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        start_time = time.time()
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
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.pipelines) if model_manager else 0
    }


# ============================================================================
# Helper Functions
# ============================================================================

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string"""
    # Simple formatting - you can customize this for specific models
    formatted = ""
    for msg in messages:
        if msg.role == "system":
            formatted += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            formatted += f"User: {msg.content}\n\n"
        elif msg.role == "assistant":
            formatted += f"Assistant: {msg.content}\n\n"
    
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

