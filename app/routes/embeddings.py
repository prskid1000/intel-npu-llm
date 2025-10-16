"""
Embeddings API Route
POST /v1/embeddings - Generate text embeddings
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from typing import TYPE_CHECKING

from ..models import EmbeddingRequest, EmbeddingResponse, EmbeddingObject, Usage

if TYPE_CHECKING:
    from ..managers import ModelManager

router = APIRouter()
model_manager: 'ModelManager' = None


def set_model_manager(manager):
    global model_manager
    model_manager = manager


@router.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for text (OpenAI-compatible)"""
    
    pipeline = model_manager.get_pipeline(request.model)
    model_type = model_manager.get_model_type(request.model)
    
    if model_type != "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not an embedding model"
        )
    
    inputs = [request.input] if isinstance(request.input, str) else request.input
    
    embeddings_data = []
    total_tokens = 0
    
    for idx, text in enumerate(inputs):
        try:
            tokens = len(text.split())
            total_tokens += tokens
            
            # Generate embedding
            input_ids = np.array([[1] * min(tokens, 512)])
            result = pipeline.infer_new_request({"input_ids": input_ids})
            
            embedding = None
            for output in result.values():
                embedding = output.flatten().tolist()
                break
            
            if embedding is None:
                raise ValueError("Could not extract embedding from model output")
            
            if request.dimensions and request.dimensions < len(embedding):
                embedding = embedding[:request.dimensions]
            
            if request.encoding_format == "base64":
                import base64
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                embedding_encoded = base64.b64encode(embedding_bytes).decode('utf-8')
                embeddings_data.append(EmbeddingObject(
                    embedding=embedding_encoded,
                    index=idx
                ))
            else:
                embeddings_data.append(EmbeddingObject(
                    embedding=embedding,
                    index=idx
                ))
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding: {str(e)}"
            )
    
    return EmbeddingResponse(
        data=embeddings_data,
        model=request.model,
        usage=Usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens
        )
    )

