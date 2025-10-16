"""
Models API Routes
GET /v1/models - List models
GET /v1/models/{model} - Get model details
"""

import time
from fastapi import APIRouter
from typing import TYPE_CHECKING

from ..models import ModelInfo, ModelListResponse
from ..utils import create_error_response

if TYPE_CHECKING:
    from ..managers import ModelManager

router = APIRouter()

# Will be set by main.py
model_manager: 'ModelManager' = None


def set_model_manager(manager):
    """Set the model manager instance"""
    global model_manager
    model_manager = manager


@router.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List available models (OpenAI-compatible)"""
    models = [
        ModelInfo(
            id=model_name,
            created=int(time.time()),
            owned_by="local",
            permission=[],
            root=model_name,
            parent=None
        )
        for model_name in model_manager.list_models()
    ]
    return ModelListResponse(data=models)


@router.get("/v1/models/{model}")
async def get_model(model: str):
    """Get specific model details (OpenAI-compatible)"""
    if model not in model_manager.list_models():
        return create_error_response(
            message=f"The model '{model}' does not exist",
            error_type="invalid_request_error",
            code="model_not_found",
            status_code=404
        )
    
    return ModelInfo(
        id=model,
        created=int(time.time()),
        owned_by="local",
        permission=[],
        root=model,
        parent=None
    )

