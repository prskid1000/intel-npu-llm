"""
Main FastAPI Application
Initializes app, middleware, routes, and manages lifecycle
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import ServerConfig, ModelConfig
from .managers import ModelManager, FileStorageManager, VectorStore
from .session_manager import SessionManager
from .utils import create_error_response

# Import route modules
from .routes import (
    models as models_route,
    chat as chat_route,
    completions as completions_route,
    audio as audio_route,
    files as files_route,
    embeddings as embeddings_route,
    moderation as moderation_route,
    vector_store as vector_store_route,
    images as images_route,
    mcp_servers as mcp_servers_route
)
from .mcp_client_manager import MCPClientManager
from .tool_service import tool_service
from .tool_executor import tool_executor
from . import realtime


# ============================================================================
# Configuration
# ============================================================================

# Default configuration
DEFAULT_CONFIG = ServerConfig(
    host="0.0.0.0",
    port=8000,
    models=[
        ModelConfig(
            name="qwen2.5-3b",
            path="models/Qwen/Qwen2.5-3B",
            device="NPU",
            type="llm"
        )
    ]
)

# Load configuration
try:
    with open("config.json", "r") as f:
        config_data = json.load(f)
        config = ServerConfig(**config_data)
except FileNotFoundError:
    config = DEFAULT_CONFIG


# ============================================================================
# Global Managers
# ============================================================================

model_manager: Optional[ModelManager] = None
file_storage: Optional[FileStorageManager] = None
vector_store: Optional[VectorStore] = None
session_manager: Optional[SessionManager] = None
mcp_manager: Optional[MCPClientManager] = None


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global model_manager, file_storage, vector_store, session_manager
    
    try:
        print("📋 Loaded configuration")
        
        # Initialize session manager
        session_manager = SessionManager(timeout_minutes=30, cleanup_interval=300)
        print(f"🔄 Session manager initialized (timeout: 30min)")
        
        # Initialize file storage
        file_storage = FileStorageManager(config.upload_dir)
        print(f"📁 File storage initialized at: {config.upload_dir}")
        
        # Initialize vector store
        vector_store = VectorStore(config.vector_store_dir)
        print(f"🗄️  Vector store initialized at: {config.vector_store_dir}")
        
        # Initialize model manager    
        model_manager = ModelManager(config)
        await model_manager.load_models()
        
        # Initialize MCP client manager
        mcp_manager = MCPClientManager()
        
        # Initialize tool executor with model manager (for image generation tool)
        tool_executor.set_model_manager(model_manager)
        
        # Initialize tool service with MCP manager
        tool_service.mcp_manager = mcp_manager
        
        # Set dependencies in route modules
        models_route.set_model_manager(model_manager)
        chat_route.set_dependencies(model_manager, file_storage, None)
        completions_route.set_model_manager(model_manager)
        audio_route.set_model_manager(model_manager)
        files_route.set_file_storage(file_storage)
        embeddings_route.set_model_manager(model_manager)
        moderation_route.set_model_manager(model_manager)
        vector_store_route.set_dependencies(model_manager, vector_store)
        images_route.set_model_manager(model_manager)
        mcp_servers_route.set_mcp_manager(mcp_manager)
        
        print(f"🚀 Server ready at http://{config.host}:{config.port}")
        print(f"📚 Available models: {', '.join(model_manager.list_models())}")
        print(f"🔧 Tool executor initialized with {len(tool_executor.tool_registry)} built-in tools")
        print(f"🔌 MCP Server Manager initialized")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        raise
    
    yield
    
    # Cleanup
    if mcp_manager:
        print("🔄 Disconnecting MCP servers...")
        await mcp_manager.disconnect_all_servers()
    
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication Middleware
# ============================================================================

REQUIRE_API_KEY = os.getenv("OPENAI_API_KEY_REQUIRED", "false").lower() == "true"
VALID_API_KEYS = set(os.getenv("OPENAI_API_KEYS", "").split(",")) if os.getenv("OPENAI_API_KEYS") else set()


@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    """Validate API key if required"""
    if request.url.path in ["/health", "/", "/docs", "/openapi.json", "/redoc"]:
        return await call_next(request)
    
    if REQUIRE_API_KEY or VALID_API_KEYS:
        auth_header = request.headers.get("authorization", "")
        
        if not auth_header:
            return create_error_response(
                message="Missing API key in Authorization header",
                error_type="invalid_request_error",
                code="missing_api_key",
                status_code=401
            )
        
        if not auth_header.startswith("Bearer "):
            return create_error_response(
                message="Invalid Authorization header format",
                error_type="invalid_request_error",
                code="invalid_api_key",
                status_code=401
            )
        
        api_key = auth_header[7:]
        
        if VALID_API_KEYS and api_key not in VALID_API_KEYS:
            return create_error_response(
                message="Invalid API key",
                error_type="invalid_request_error",
                code="invalid_api_key",
                status_code=401
            )
    
    response = await call_next(request)
    return response


# ============================================================================
# Register Routes
# ============================================================================

# Include all routers
app.include_router(models_route.router, tags=["Models"])
app.include_router(chat_route.router, tags=["Chat"])
app.include_router(completions_route.router, tags=["Completions"])
app.include_router(audio_route.router, tags=["Audio"])
app.include_router(files_route.router, tags=["Files"])
app.include_router(embeddings_route.router, tags=["Embeddings"])
app.include_router(moderation_route.router, tags=["Moderation"])
app.include_router(vector_store_route.router, tags=["Vector Store"])
app.include_router(images_route.router, tags=["Images"])
app.include_router(mcp_servers_route.router, tags=["MCP Servers"])


# ============================================================================
# Additional Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenVINO GenAI Server",
        "version": "1.0.0",
        "models": model_manager.list_models() if model_manager else [],
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    llm_count = len(model_manager.llm_pipelines) if model_manager else 0
    vlm_count = len(model_manager.vlm_pipelines) if model_manager else 0
    whisper_count = len(model_manager.whisper_pipelines) if model_manager else 0
    tts_count = len(model_manager.tts_pipelines) if model_manager else 0
    embedding_count = len(model_manager.embedding_pipelines) if model_manager else 0
    text2image_count = len(model_manager.text2image_pipelines) if model_manager else 0
    moderation_count = len(model_manager.moderation_pipelines) if model_manager else 0
    
    return {
        "status": "healthy",
        "models_loaded": llm_count + vlm_count + whisper_count + tts_count + embedding_count + text2image_count + moderation_count,
        "llm_models": llm_count,
        "vlm_models": vlm_count,
        "whisper_models": whisper_count,
        "tts_models": tts_count,
        "embedding_models": embedding_count,
        "text2image_models": text2image_count,
        "moderation_models": moderation_count,
        "files_stored": len(file_storage.files_metadata) if file_storage else 0,
        "documents_in_vector_store": len(vector_store.vectors) if vector_store else 0
    }


@app.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket, model: str = "qwen2.5-3b"):
    """WebSocket Realtime API endpoint"""
    await realtime.realtime_endpoint(websocket, model, model_manager, session_manager)


# Session Management REST API
@app.get("/v1/realtime/sessions")
async def list_realtime_sessions():
    """List all active realtime sessions"""
    return await realtime.list_sessions(session_manager)


@app.get("/v1/realtime/sessions/{session_id}")
async def get_realtime_session(session_id: str):
    """Get details of a specific realtime session"""
    return await realtime.get_session(session_id, session_manager)


@app.delete("/v1/realtime/sessions/{session_id}")
async def delete_realtime_session(session_id: str):
    """Delete a realtime session"""
    return await realtime.delete_session(session_id, session_manager)


