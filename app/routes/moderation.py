"""
Moderation API Route
POST /v1/moderations - Content moderation
"""

import uuid
from fastapi import APIRouter, HTTPException
from typing import TYPE_CHECKING

from ..models import (
    ModerationRequest,
    ModerationResponse,
    ModerationResult,
    ModerationCategories,
    ModerationCategoryScores
)

if TYPE_CHECKING:
    from ..managers import ModelManager

router = APIRouter()
model_manager: 'ModelManager' = None


def set_model_manager(manager):
    global model_manager
    model_manager = manager


@router.post("/v1/moderations")
async def create_moderation(request: ModerationRequest) -> ModerationResponse:
    """Moderate text content (OpenAI-compatible)"""
    
    model_name = request.model if request.model else None
    if not model_name:
        moderation_models = list(model_manager.moderation_pipelines.keys())
        if not moderation_models:
            raise HTTPException(
                status_code=400,
                detail="No moderation models available"
            )
        model_name = moderation_models[0]
    
    pipeline = model_manager.get_pipeline(model_name)
    model_type = model_manager.get_model_type(model_name)
    
    if model_type != "moderation":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not a moderation model"
        )
    
    inputs = [request.input] if isinstance(request.input, str) else request.input
    results = []
    
    for text in inputs:
        try:
            input_data = {"text": text}
            result = pipeline.infer_new_request(input_data)
            
            categories = ModerationCategories()
            category_scores = ModerationCategoryScores()
            flagged = False
            
            for key, value in result.items():
                scores = value.flatten().tolist()
                if len(scores) >= 11:
                    category_scores.hate = float(scores[0])
                    category_scores.hate_threatening = float(scores[1])
                    category_scores.harassment = float(scores[2])
                    category_scores.harassment_threatening = float(scores[3])
                    category_scores.self_harm = float(scores[4])
                    category_scores.self_harm_intent = float(scores[5])
                    category_scores.self_harm_instructions = float(scores[6])
                    category_scores.sexual = float(scores[7])
                    category_scores.sexual_minors = float(scores[8])
                    category_scores.violence = float(scores[9])
                    category_scores.violence_graphic = float(scores[10])
                    
                    threshold = 0.5
                    categories.hate = scores[0] > threshold
                    categories.hate_threatening = scores[1] > threshold
                    categories.harassment = scores[2] > threshold
                    categories.harassment_threatening = scores[3] > threshold
                    categories.self_harm = scores[4] > threshold
                    categories.self_harm_intent = scores[5] > threshold
                    categories.self_harm_instructions = scores[6] > threshold
                    categories.sexual = scores[7] > threshold
                    categories.sexual_minors = scores[8] > threshold
                    categories.violence = scores[9] > threshold
                    categories.violence_graphic = scores[10] > threshold
                    
                    flagged = any([
                        categories.hate, categories.hate_threatening,
                        categories.harassment, categories.harassment_threatening,
                        categories.self_harm, categories.self_harm_intent,
                        categories.self_harm_instructions, categories.sexual,
                        categories.sexual_minors, categories.violence,
                        categories.violence_graphic
                    ])
                break
            
            results.append(ModerationResult(
                flagged=flagged,
                categories=categories,
                category_scores=category_scores
            ))
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Moderation failed: {str(e)}"
            )
    
    return ModerationResponse(
        id=f"modr-{uuid.uuid4().hex[:8]}",
        model=model_name,
        results=results
    )

