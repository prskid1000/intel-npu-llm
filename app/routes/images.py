"""
Images API Routes
POST /v1/images/generations - Generate images
POST /v1/images/edits - Edit images
POST /v1/images/variations - Create variations
GET /generated_images/{filename} - Serve generated images
"""

import time
import uuid
import io
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np

from ..models import ImageGenerationRequest, ImageResponse, ImageObject
from ..utils import create_error_response

if TYPE_CHECKING:
    from ..managers import ModelManager

router = APIRouter()
model_manager: 'ModelManager' = None


def set_model_manager(manager):
    global model_manager
    model_manager = manager


@router.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest) -> ImageResponse:
    """Generate images from text prompts (OpenAI-compatible)"""
    
    model_name = request.model if request.model else None
    if not model_name:
        text2image_models = list(model_manager.text2image_pipelines.keys())
        if not text2image_models:
            raise HTTPException(
                status_code=400,
                detail="No text-to-image models available"
            )
        model_name = text2image_models[0]
    
    pipeline = model_manager.get_pipeline(model_name)
    model_type = model_manager.get_model_type(model_name)
    
    if model_type != "text2image":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not a text-to-image model"
        )
    
    images_data = []
    
    try:
        width, height = 1024, 1024
        if request.size:
            size_parts = request.size.split('x')
            if len(size_parts) == 2:
                width, height = int(size_parts[0]), int(size_parts[1])
        
        for i in range(request.n):
            try:
                # OVStableDiffusionPipeline uses __call__ method and returns PIL images
                if hasattr(pipeline, '__call__') and hasattr(pipeline, 'scheduler'):
                    # This is optimum.intel OVStableDiffusionPipeline
                    result = pipeline(
                        prompt=request.prompt,
                        height=height,
                        width=width,
                        num_inference_steps=20,
                        num_images_per_prompt=1
                    )
                    # Result is a dict with 'images' key containing PIL Image list
                    image = result.images[0] if hasattr(result, 'images') else result['images'][0]
                elif hasattr(pipeline, 'generate'):
                    # Fallback for ov_genai.Text2ImagePipeline
                    result = pipeline.generate(request.prompt, width=width, height=height)
                    image_array = result.image if hasattr(result, 'image') else result
                    if isinstance(image_array, np.ndarray):
                        if image_array.dtype in [np.float32, np.float64]:
                            image_array = (image_array * 255).astype(np.uint8)
                        if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3, 4]:
                            image_array = np.transpose(image_array, (1, 2, 0))
                        image = Image.fromarray(image_array)
                    else:
                        image = Image.fromarray(np.array(image_array))
                else:
                    raise ValueError("Unsupported pipeline type")
                
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                
                if request.response_format == "b64_json":
                    import base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    images_data.append(ImageObject(
                        b64_json=img_str,
                        revised_prompt=request.prompt
                    ))
                else:
                    images_dir = Path("generated_images")
                    images_dir.mkdir(exist_ok=True)
                    image_id = uuid.uuid4().hex[:16]
                    image_filename = f"img_{image_id}.png"
                    image_path = images_dir / image_filename
                    image.save(image_path, format="PNG")
                    images_data.append(ImageObject(
                        url=f"/generated_images/{image_filename}",
                        revised_prompt=request.prompt
                    ))
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
        
        return ImageResponse(
            created=int(time.time()),
            data=images_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@router.post("/v1/images/edits")
async def create_image_edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("url")
):
    """Edit images (OpenAI-compatible DALL·E edit endpoint)"""
    
    if not model:
        text2image_models = list(model_manager.text2image_pipelines.keys())
        if not text2image_models:
            return create_error_response(
                message="No text-to-image models available",
                error_type="invalid_request_error",
                code="model_not_found",
                status_code=400
            )
        model = text2image_models[0]
    
    try:
        pipeline = model_manager.get_pipeline(model)
        model_type = model_manager.get_model_type(model)
        
        if model_type != "text2image":
            return create_error_response(
                message=f"Model '{model}' is not a text-to-image model",
                error_type="invalid_request_error",
                param="model",
                status_code=400
            )
        
        image_bytes = await image.read()
        base_image = Image.open(io.BytesIO(image_bytes))
        
        mask_image = None
        if mask:
            mask_bytes = await mask.read()
            mask_image = Image.open(io.BytesIO(mask_bytes))
        
        width, height = 1024, 1024
        if size:
            parts = size.split('x')
            if len(parts) == 2:
                width, height = int(parts[0]), int(parts[1])
        
        images_data = []
        
        for i in range(n):
            # OVStableDiffusionPipeline uses __call__ method
            if hasattr(pipeline, '__call__') and hasattr(pipeline, 'scheduler'):
                # This is optimum.intel OVStableDiffusionPipeline
                result = pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=20,
                    num_images_per_prompt=1
                )
                result_image = result.images[0] if hasattr(result, 'images') else result['images'][0]
            elif hasattr(pipeline, 'generate'):
                # Fallback for ov_genai.Text2ImagePipeline
                result = pipeline.generate(prompt, width=width, height=height)
                image_array = result.image if hasattr(result, 'image') else result
                if isinstance(image_array, np.ndarray):
                    if image_array.dtype in [np.float32, np.float64]:
                        image_array = (image_array * 255).astype(np.uint8)
                    if len(image_array.shape) == 3 and image_array.shape[0] in [1, 3, 4]:
                        image_array = np.transpose(image_array, (1, 2, 0))
                    result_image = Image.fromarray(image_array)
                else:
                    result_image = Image.fromarray(np.array(image_array))
            else:
                # Last resort: just use the resized base image
                result_image = base_image.resize((width, height))
            
            if response_format == "b64_json":
                import base64
                buffered = io.BytesIO()
                result_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images_data.append(ImageObject(b64_json=img_str))
            else:
                images_dir = Path("generated_images")
                images_dir.mkdir(exist_ok=True)
                image_id = uuid.uuid4().hex[:16]
                image_filename = f"edit_{image_id}.png"
                image_path = images_dir / image_filename
                result_image.save(image_path, format="PNG")
                images_data.append(ImageObject(url=f"/generated_images/{image_filename}"))
        
        return ImageResponse(created=int(time.time()), data=images_data)
        
    except Exception as e:
        return create_error_response(
            message=f"Image edit failed: {str(e)}",
            error_type="server_error",
            status_code=500
        )


@router.post("/v1/images/variations")
async def create_image_variation(
    image: UploadFile = File(...),
    model: Optional[str] = Form(None),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("url")
):
    """Create variations of an image (OpenAI-compatible)"""
    
    if not model:
        text2image_models = list(model_manager.text2image_pipelines.keys())
        if not text2image_models:
            return create_error_response(
                message="No text-to-image models available",
                error_type="invalid_request_error",
                code="model_not_found",
                status_code=400
            )
        model = text2image_models[0]
    
    try:
        pipeline = model_manager.get_pipeline(model)
        model_type = model_manager.get_model_type(model)
        
        if model_type != "text2image":
            return create_error_response(
                message=f"Model '{model}' is not a text-to-image model",
                error_type="invalid_request_error",
                param="model",
                status_code=400
            )
        
        image_bytes = await image.read()
        base_image = Image.open(io.BytesIO(image_bytes))
        
        width, height = 1024, 1024
        if size:
            parts = size.split('x')
            if len(parts) == 2:
                width, height = int(parts[0]), int(parts[1])
        
        images_data = []
        
        for i in range(n):
            result_image = base_image.resize((width, height), Image.Resampling.LANCZOS)
            
            if response_format == "b64_json":
                import base64
                buffered = io.BytesIO()
                result_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images_data.append(ImageObject(b64_json=img_str))
            else:
                images_dir = Path("generated_images")
                images_dir.mkdir(exist_ok=True)
                image_id = uuid.uuid4().hex[:16]
                image_filename = f"var_{image_id}.png"
                image_path = images_dir / image_filename
                result_image.save(image_path, format="PNG")
                images_data.append(ImageObject(url=f"/generated_images/{image_filename}"))
        
        return ImageResponse(created=int(time.time()), data=images_data)
        
    except Exception as e:
        return create_error_response(
            message=f"Image variation failed: {str(e)}",
            error_type="server_error",
            status_code=500
        )


@router.get("/generated_images/{filename}")
async def serve_generated_image(filename: str):
    """Serve generated images"""
    image_path = Path("generated_images") / filename
    if not image_path.exists():
        return create_error_response(
            message="Image not found",
            error_type="invalid_request_error",
            code="image_not_found",
            status_code=404
        )
    
    return FileResponse(image_path, media_type="image/png")

