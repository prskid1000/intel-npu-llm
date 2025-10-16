"""
Vector Store API Routes
POST /v1/vector_store/documents - Add document
POST /v1/vector_store/search - Search documents
GET /v1/vector_store/documents - List documents
GET /v1/vector_store/documents/{id} - Get document
DELETE /v1/vector_store/documents/{id} - Delete document
POST /v1/vector_store/clear - Clear all
"""

import uuid
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import TYPE_CHECKING

from ..models import VectorStoreRequest, VectorSearchRequest

if TYPE_CHECKING:
    from ..managers import ModelManager, VectorStore

router = APIRouter()
model_manager: 'ModelManager' = None
vector_store: 'VectorStore' = None


def set_dependencies(mm, vs):
    global model_manager, vector_store
    model_manager = mm
    vector_store = vs


@router.post("/v1/vector_store/documents")
async def add_document_to_vector_store(request: VectorStoreRequest):
    """Add a document to the vector store with its embedding"""
    
    print(f"📚 VectorStore Add: text_len={len(request.text)}, metadata={bool(request.metadata)}")
    
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    pipeline = model_manager.get_pipeline(request.embedding_model)
    model_type = model_manager.get_model_type(request.embedding_model)
    
    if model_type != "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.embedding_model}' is not an embedding model"
        )
    
    try:
        # Properly tokenize using transformers
        from transformers import AutoTokenizer
        model_config = model_manager.model_configs.get(request.embedding_model)
        if not model_config:
            raise ValueError(f"Model config for '{request.embedding_model}' not found")
        
        tokenizer = AutoTokenizer.from_pretrained(model_config.path)
        encoded = tokenizer(request.text, padding=True, truncation=True, max_length=512, return_tensors="np")
        
        result = pipeline.infer_new_request(dict(encoded))
        
        embedding = None
        for output in result.values():
            # Apply mean pooling for sentence transformers (BGE model)
            if len(output.shape) == 3:
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
                # Fallback: use output as-is
                embedding = output.flatten().tolist()
            break
        
        if embedding is None:
            raise ValueError("Could not generate embedding")
        
        doc_id = f"doc-{uuid.uuid4().hex[:16]}"
        
        vector_store.add_document(
            doc_id=doc_id,
            text=request.text,
            embedding=embedding,
            metadata=request.metadata
        )
        
        return {
            "doc_id": doc_id,
            "status": "added",
            "text_length": len(request.text),
            "embedding_dimension": len(embedding)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add document: {str(e)}"
        )


@router.post("/v1/vector_store/search")
async def search_vector_store(request: VectorSearchRequest):
    """Search the vector store for similar documents"""
    
    print(f"🔍 VectorStore Search: query_len={len(request.query)}, k={request.top_k}")
    
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    pipeline = model_manager.get_pipeline(request.embedding_model)
    model_type = model_manager.get_model_type(request.embedding_model)
    
    if model_type != "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.embedding_model}' is not an embedding model"
        )
    
    try:
        # Properly tokenize using transformers
        from transformers import AutoTokenizer
        model_config = model_manager.model_configs.get(request.embedding_model)
        if not model_config:
            raise ValueError(f"Model config for '{request.embedding_model}' not found")
        
        tokenizer = AutoTokenizer.from_pretrained(model_config.path)
        encoded = tokenizer(request.query, padding=True, truncation=True, max_length=512, return_tensors="np")
        
        result = pipeline.infer_new_request(dict(encoded))
        
        query_embedding = None
        for output in result.values():
            # Apply mean pooling for sentence transformers (BGE model)
            if len(output.shape) == 3:
                attention_mask = encoded.get('attention_mask', None)
                if attention_mask is not None:
                    # Masked mean pooling
                    attention_mask_expanded = np.expand_dims(attention_mask, -1)
                    sum_embeddings = np.sum(output * attention_mask_expanded, axis=1)
                    sum_mask = np.clip(attention_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
                    query_embedding = (sum_embeddings / sum_mask)[0].tolist()
                else:
                    # Simple mean pooling
                    query_embedding = output.mean(axis=1)[0].tolist()
            else:
                # Fallback: use output as-is
                query_embedding = output.flatten().tolist()
            break
        
        if query_embedding is None:
            raise ValueError("Could not generate query embedding")
        
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/v1/vector_store/documents")
async def list_vector_documents():
    """List all documents in the vector store"""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    documents = vector_store.list_documents()
    return {
        "documents": documents,
        "count": len(documents)
    }


@router.get("/v1/vector_store/documents/{doc_id}")
async def get_vector_document(doc_id: str):
    """Get a specific document from the vector store"""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    document = vector_store.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    
    return {
        "doc_id": doc_id,
        "text": document["text"],
        "metadata": document["metadata"],
        "created_at": document["created_at"],
        "embedding_dimension": len(document["embedding"])
    }


@router.delete("/v1/vector_store/documents/{doc_id}")
async def delete_vector_document(doc_id: str):
    """Delete a document from the vector store"""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    deleted = vector_store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    
    return {
        "doc_id": doc_id,
        "deleted": True
    }


@router.post("/v1/vector_store/clear")
async def clear_vector_store():
    """Clear all documents from the vector store"""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    vector_store.clear()
    return {
        "status": "cleared",
        "message": "All documents have been removed from the vector store"
    }

