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
    
    # Log request
    inputs_count = 1 if isinstance(request.input, str) else len(request.input)
    print(f"📊 Embeddings: model={request.model}, inputs={inputs_count}")
    
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
    
    # Load tokenizer once outside the loop
    from transformers import AutoTokenizer
    model_config = model_manager.model_configs.get(request.model)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model config for '{request.model}' not found")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    
    for idx, text in enumerate(inputs):
        try:
            # Tokenize
            encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="np")
            tokens = len(encoded['input_ids'][0])
            total_tokens += tokens
            
            # Generate embedding using the properly tokenized inputs
            result = pipeline.infer_new_request(dict(encoded))
            
            embedding = None
            for output in result.values():
                # Apply mean pooling for sentence transformers (BGE model)
                # Output shape is typically (batch_size, seq_length, hidden_size)
                if len(output.shape) == 3:
                    # Mean pooling: average across sequence dimension
                    attention_mask = encoded.get('attention_mask', None)
                    if attention_mask is not None:
                        # Masked mean pooling
                        attention_mask_expanded = np.expand_dims(attention_mask, -1)
                        sum_embeddings = np.sum(output * attention_mask_expanded, axis=1)
                        sum_mask = np.clip(attention_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
                        embedding = (sum_embeddings / sum_mask)[0].tolist()
                    else:
                        # Simple mean pooling
                        embedding = output.mean(axis=1)[0].tolist()
                else:
                    # Fallback: use output as-is (e.g., if already pooled)
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

