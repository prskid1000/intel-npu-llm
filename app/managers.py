"""
Model and Resource Managers
Handles model loading, file storage, and vector store operations
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

import openvino_genai as ov_genai
import openvino as ov
from fastapi import HTTPException

from .models import ModelConfig, ServerConfig, FileObject


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages multiple OpenVINO GenAI pipelines"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.llm_pipelines: Dict[str, ov_genai.LLMPipeline] = {}
        self.vlm_pipelines: Dict[str, ov_genai.VLMPipeline] = {}
        self.whisper_pipelines: Dict[str, ov_genai.WhisperPipeline] = {}
        self.tts_pipelines: Dict[str, ov_genai.Text2SpeechPipeline] = {}
        self.embedding_pipelines: Dict[str, Any] = {}
        self.text2image_pipelines: Dict[str, Any] = {}
        self.moderation_pipelines: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        
    async def load_models(self):
        """Load all configured models"""
        print("🔄 Loading models...")
        for model_config in self.config.models:
            try:
                print(f"  Loading {model_config.name} ({model_config.type}) on {model_config.device}...")
                
                if model_config.type == "llm":
                    pipeline = ov_genai.LLMPipeline(model_config.path, model_config.device)
                    self.llm_pipelines[model_config.name] = pipeline
                
                elif model_config.type == "vlm":
                    pipeline = ov_genai.VLMPipeline(model_config.path, model_config.device)
                    self.vlm_pipelines[model_config.name] = pipeline
                
                elif model_config.type == "whisper":
                    pipeline = ov_genai.WhisperPipeline(model_config.path, model_config.device)
                    self.whisper_pipelines[model_config.name] = pipeline
                
                elif model_config.type == "tts":
                    pipeline = ov_genai.Text2SpeechPipeline(model_config.path, model_config.device)
                    self.tts_pipelines[model_config.name] = pipeline
                
                elif model_config.type == "embedding":
                    core = ov.Core()
                    model_path = Path(model_config.path) / "openvino_model.xml"
                    model = core.read_model(model=str(model_path))
                    compiled_model = core.compile_model(model, model_config.device)
                    self.embedding_pipelines[model_config.name] = compiled_model
                
                elif model_config.type == "text2image":
                    try:
                        # Use optimum.intel.OVPipelineForText2Image instead of ov_genai.Text2ImagePipeline
                        # to avoid dtype compatibility issues with tokenizers
                        from optimum.intel import OVStableDiffusionPipeline
                        pipeline = OVStableDiffusionPipeline.from_pretrained(
                            model_config.path,
                            device=model_config.device,
                            compile=True
                        )
                        self.text2image_pipelines[model_config.name] = pipeline
                    except Exception as e:
                        print(f"⚠️  Warning: Could not load Text2Image pipeline: {e}")
                        print(f"   Trying fallback method...")
                        core = ov.Core()
                        model_path = Path(model_config.path) / "openvino_model.xml"
                        model = core.read_model(model=str(model_path))
                        compiled_model = core.compile_model(model, model_config.device)
                        self.text2image_pipelines[model_config.name] = compiled_model
                
                elif model_config.type == "moderation":
                    core = ov.Core()
                    model_path = Path(model_config.path) / "openvino_model.xml"
                    model = core.read_model(model=str(model_path))
                    compiled_model = core.compile_model(model, model_config.device)
                    self.moderation_pipelines[model_config.name] = compiled_model
                
                else:
                    print(f"  ⚠️  Unknown model type: {model_config.type}")
                    continue
                
                self.model_configs[model_config.name] = model_config
                print(f"  ✅ {model_config.name} loaded successfully")
                
            except Exception as e:
                print(f"  ❌ Failed to load {model_config.name}: {e}")
                
        total_models = (len(self.llm_pipelines) + len(self.vlm_pipelines) + 
                       len(self.whisper_pipelines) + len(self.tts_pipelines) +
                       len(self.embedding_pipelines) + len(self.text2image_pipelines) +
                       len(self.moderation_pipelines))
        if total_models == 0:
            raise RuntimeError("No models loaded successfully!")
            
        print(f"✅ Loaded {total_models} model(s)")
        
    def reload_vlm_pipeline(self, model_name: str) -> bool:
        """Reload a VLM pipeline (workaround for NPU history accumulation)"""
        if model_name not in self.model_configs:
            return False
        
        model_config = self.model_configs[model_name]
        if model_config.type != "vlm":
            return False
        
        try:
            if model_config.device == "NPU":
                # NPU requires full pipeline reload (offload + reload)
                print(f"🔄 NPU VLM: Offloading and reloading pipeline {model_name} (clearing history)")
                # Delete old pipeline to free NPU memory
                if model_name in self.vlm_pipelines:
                    del self.vlm_pipelines[model_name]
                    # Recreate pipeline from scratch
                    pipeline = ov_genai.VLMPipeline(model_config.path, model_config.device)
                    self.vlm_pipelines[model_name] = pipeline
                    print(f"✅ NPU VLM reloaded successfully")
            else:
                # CPU/GPU can use finish_chat() to reset history
                print(f"🔄 {model_config.device} VLM: Resetting conversation history for {model_name}")
                pipeline = self.vlm_pipelines.get(model_name)
                if pipeline and hasattr(pipeline, 'finish_chat'):
                    pipeline.finish_chat()
                    print(f"✅ Conversation history reset via finish_chat()")
                elif pipeline and hasattr(pipeline, 'start_chat'):
                    pipeline.start_chat()
                    print(f"✅ Conversation history reset via start_chat()")
                else:
                    print(f"⚠️  No reset method available, continuing...")
            return True
        except Exception as e:
            print(f"⚠️  Failed to reload VLM pipeline: {e}")
            return False
    
    def get_pipeline(self, model_name: str) -> Any:
        """Get a pipeline by model name"""
        if model_name in self.llm_pipelines:
            return self.llm_pipelines[model_name]
        elif model_name in self.vlm_pipelines:
            # No proactive reload - let the error trigger the reload in chat.py
            return self.vlm_pipelines[model_name]
        elif model_name in self.whisper_pipelines:
            return self.whisper_pipelines[model_name]
        elif model_name in self.tts_pipelines:
            return self.tts_pipelines[model_name]
        elif model_name in self.embedding_pipelines:
            return self.embedding_pipelines[model_name]
        elif model_name in self.text2image_pipelines:
            return self.text2image_pipelines[model_name]
        elif model_name in self.moderation_pipelines:
            return self.moderation_pipelines[model_name]
        else:
            all_models = (list(self.llm_pipelines.keys()) + list(self.vlm_pipelines.keys()) +
                         list(self.whisper_pipelines.keys()) + list(self.tts_pipelines.keys()) +
                         list(self.embedding_pipelines.keys()) + list(self.text2image_pipelines.keys()) +
                         list(self.moderation_pipelines.keys()))
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {all_models}"
            )
    
    def get_model_type(self, model_name: str) -> str:
        """Get the type of a model"""
        if model_name in self.model_configs:
            return self.model_configs[model_name].type
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return (list(self.llm_pipelines.keys()) + list(self.vlm_pipelines.keys()) +
                list(self.whisper_pipelines.keys()) + list(self.tts_pipelines.keys()) +
                list(self.embedding_pipelines.keys()) + list(self.text2image_pipelines.keys()) +
                list(self.moderation_pipelines.keys()))
    
    async def cleanup(self):
        """Cleanup all loaded models"""
        print("🔄 Cleaning up models...")
        self.llm_pipelines.clear()
        self.vlm_pipelines.clear()
        self.whisper_pipelines.clear()
        self.tts_pipelines.clear()
        self.embedding_pipelines.clear()
        self.model_configs.clear()
        print("✅ Cleanup complete")


# ============================================================================
# File Storage Manager
# ============================================================================

class FileStorageManager:
    """Manages uploaded files and their metadata"""
    
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.files_metadata: Dict[str, FileObject] = {}
        self.metadata_file = self.upload_dir / "files_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load file metadata from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.files_metadata = {k: FileObject(**v) for k, v in data.items()}
    
    def _save_metadata(self):
        """Save file metadata to disk"""
        data = {k: v.model_dump() for k, v in self.files_metadata.items()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def save_file(self, file, purpose: str = "assistants") -> FileObject:
        """Save uploaded file and return metadata"""
        file_id = f"file-{uuid.uuid4().hex}"
        file_path = self.upload_dir / f"{file_id}_{file.filename}"
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        file_obj = FileObject(
            id=file_id,
            bytes=len(content),
            created_at=int(time.time()),
            filename=file.filename,
            purpose=purpose
        )
        
        self.files_metadata[file_id] = file_obj
        self._save_metadata()
        
        return file_obj
    
    def get_file(self, file_id: str) -> Optional[FileObject]:
        """Get file metadata"""
        return self.files_metadata.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """Get physical path to file"""
        file_obj = self.get_file(file_id)
        if not file_obj:
            return None
        
        for file_path in self.upload_dir.glob(f"{file_id}_*"):
            return file_path
        return None
    
    def list_files(self) -> List[FileObject]:
        """List all uploaded files"""
        return list(self.files_metadata.values())
    
    def delete_file(self, file_id: str) -> bool:
        """Delete file and its metadata"""
        file_path = self.get_file_path(file_id)
        if file_path and file_path.exists():
            file_path.unlink()
        
        if file_id in self.files_metadata:
            del self.files_metadata[file_id]
            self._save_metadata()
            return True
        return False


# ============================================================================
# Vector Store
# ============================================================================

class VectorStore:
    """Simple vector store for document embeddings and similarity search"""
    
    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        self.index_file = self.store_dir / "vector_index.json"
        self.vectors: Dict[str, Dict[str, Any]] = {}
        self._load_index()
    
    def _load_index(self):
        """Load vector index from disk"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                data = json.load(f)
                self.vectors = data
    
    def _save_index(self):
        """Save vector index to disk"""
        with open(self.index_file, 'w') as f:
            json.dump(self.vectors, f, indent=2)
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a document and its embedding to the store"""
        self.vectors[doc_id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": int(time.time())
        }
        self._save_index()
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        if not self.vectors:
            return []
        
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        similarities = []
        
        for doc_id, doc_data in self.vectors.items():
            # Skip entries that are not dictionaries (e.g., legacy list entries)
            if not isinstance(doc_data, dict):
                continue
            
            # Skip entries that don't have required fields
            if "embedding" not in doc_data or "text" not in doc_data:
                continue
            
            doc_vec = np.array(doc_data["embedding"])
            doc_norm = np.linalg.norm(doc_vec)
            
            if doc_norm == 0:
                continue
            
            similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
            
            if similarity >= threshold:
                similarities.append({
                    "doc_id": doc_id,
                    "text": doc_data["text"],
                    "similarity": float(similarity),
                    "metadata": doc_data.get("metadata", {})
                })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store"""
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            self._save_index()
            return True
        return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        return self.vectors.get(doc_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents"""
        return [
            {
                "doc_id": doc_id,
                "text": data["text"][:100] + "..." if len(data["text"]) > 100 else data["text"],
                "metadata": data.get("metadata", {}),
                "created_at": data.get("created_at", 0)
            }
            for doc_id, data in self.vectors.items()
            if isinstance(data, dict) and "text" in data
        ]
    
    def clear(self):
        """Clear all documents"""
        self.vectors.clear()
        self._save_index()


# ============================================================================
# Document Processor
# ============================================================================

class DocumentProcessor:
    """Extract text from various document formats"""
    
    @staticmethod
    def extract_text(file_path: Path) -> str:
        """Extract text from file based on extension"""
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return f"[Image: {file_path.name}]"
        
        elif extension == '.pdf':
            try:
                import PyPDF2
                text = []
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text.append(page.extract_text())
                return '\n'.join(text)
            except ImportError:
                return "[PDF file - PyPDF2 not installed for text extraction]"
        
        elif extension in ['.doc', '.docx']:
            try:
                import docx
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                return "[Word document - python-docx not installed for text extraction]"
        
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return f"[Unsupported file type: {extension}]"

