"""API Routes Package"""

from fastapi import APIRouter

# Import all routers
from .models import router as models_router
from .chat import router as chat_router  
from .audio import router as audio_router
from .images import router as images_router
from .files import router as files_router
from .embeddings import router as embeddings_router
from .moderation import router as moderation_router
from .vector_store import router as vector_store_router

__all__ = [
    "models_router",
    "chat_router",
    "audio_router",
    "images_router",
    "files_router",
    "embeddings_router",
    "moderation_router",
    "vector_store_router",
]

