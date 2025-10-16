"""
Files API Routes
POST /v1/files - Upload file
GET /v1/files - List files
GET /v1/files/{id} - Get file info
DELETE /v1/files/{id} - Delete file
GET /v1/files/{id}/content - Download file
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from typing import TYPE_CHECKING

from ..models import FileObject, FileListResponse, FileDeleteResponse

if TYPE_CHECKING:
    from ..managers import FileStorageManager

router = APIRouter()
file_storage: 'FileStorageManager' = None


def set_file_storage(fs):
    global file_storage
    file_storage = fs


@router.post("/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("assistants")
) -> FileObject:
    """Upload a file (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    file_obj = await file_storage.save_file(file, purpose)
    return file_obj


@router.get("/v1/files")
async def list_files() -> FileListResponse:
    """List all uploaded files (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    files = file_storage.list_files()
    return FileListResponse(data=files)


@router.get("/v1/files/{file_id}")
async def get_file(file_id: str) -> FileObject:
    """Get file metadata (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    file_obj = file_storage.get_file(file_id)
    if not file_obj:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    
    return file_obj


@router.delete("/v1/files/{file_id}")
async def delete_file(file_id: str) -> FileDeleteResponse:
    """Delete a file (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    deleted = file_storage.delete_file(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    
    return FileDeleteResponse(id=file_id, deleted=True)


@router.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str):
    """Get file content (OpenAI-compatible)"""
    if not file_storage:
        raise HTTPException(status_code=500, detail="File storage not initialized")
    
    file_path = file_storage.get_file_path(file_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    
    return FileResponse(file_path)

