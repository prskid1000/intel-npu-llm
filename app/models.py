"""
Pydantic Models for OpenAI API Compatibility
All request/response schemas
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


# ============================================================================
# Configuration Models
# ============================================================================

class ModelConfig(BaseModel):
    """Configuration for a single model"""
    name: str
    path: str
    device: str = "NPU"  # NPU, CPU, or GPU
    type: str = "llm"  # llm, vlm, embedding, whisper, tts, text2image, or moderation


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    models: List[ModelConfig]
    upload_dir: str = "uploads"
    vector_store_dir: str = "vector_store"


# ============================================================================
# Message Content Models
# ============================================================================

class ImageUrl(BaseModel):
    """Image URL or base64 data"""
    url: str
    detail: Optional[str] = "auto"


class MessageContent(BaseModel):
    """Message content part (text or image)"""
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


# ============================================================================
# Tool Calling Models
# ============================================================================

class FunctionCall(BaseModel):
    """Function call in a message"""
    name: str
    arguments: str  # JSON string of arguments


class ToolCall(BaseModel):
    """Tool call in a message"""
    id: str
    type: str = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    """Function definition for tool calling"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    """Tool definition"""
    type: str = "function"
    function: FunctionDefinition


# ============================================================================
# Chat Models
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message with support for multimodal content and tool calls"""
    role: str
    content: Optional[Union[str, List[MessageContent]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ResponseFormat(BaseModel):
    """Response format specification for structured outputs"""
    type: str  # "text", "json_object", "json_schema"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    parallel_tool_calls: Optional[bool] = True


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# ============================================================================
# Completion Models
# ============================================================================

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# ============================================================================
# Model Info Models
# ============================================================================

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "openvino"
    permission: List[Any] = []
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# File Models
# ============================================================================

class FileObject(BaseModel):
    """Uploaded file object"""
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileListResponse(BaseModel):
    object: str = "list"
    data: List[FileObject]


class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool


# ============================================================================
# Audio Models
# ============================================================================

class AudioTranscriptionRequest(BaseModel):
    """Audio transcription request"""
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0


class AudioTranscriptionResponse(BaseModel):
    """Audio transcription response"""
    text: str


class AudioSpeechRequest(BaseModel):
    """Text-to-speech request"""
    model: str
    input: str
    voice: Optional[str] = "alloy"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


# ============================================================================
# Embedding Models
# ============================================================================

class EmbeddingRequest(BaseModel):
    """Embedding request"""
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    """Embedding object"""
    object: str = "embedding"
    embedding: Union[List[float], str]
    index: int


class EmbeddingResponse(BaseModel):
    """Embedding response"""
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Usage


# ============================================================================
# Moderation Models
# ============================================================================

class ModerationRequest(BaseModel):
    """Moderation request"""
    input: Union[str, List[str]]
    model: Optional[str] = "text-moderation-latest"


class ModerationCategories(BaseModel):
    """Moderation categories"""
    hate: bool = False
    hate_threatening: bool = False
    harassment: bool = False
    harassment_threatening: bool = False
    self_harm: bool = False
    self_harm_intent: bool = False
    self_harm_instructions: bool = False
    sexual: bool = False
    sexual_minors: bool = False
    violence: bool = False
    violence_graphic: bool = False


class ModerationCategoryScores(BaseModel):
    """Moderation category scores"""
    hate: float = 0.0
    hate_threatening: float = 0.0
    harassment: float = 0.0
    harassment_threatening: float = 0.0
    self_harm: float = 0.0
    self_harm_intent: float = 0.0
    self_harm_instructions: float = 0.0
    sexual: float = 0.0
    sexual_minors: float = 0.0
    violence: float = 0.0
    violence_graphic: float = 0.0


class ModerationResult(BaseModel):
    """Moderation result"""
    flagged: bool
    categories: ModerationCategories
    category_scores: ModerationCategoryScores


class ModerationResponse(BaseModel):
    """Moderation response"""
    id: str
    model: str
    results: List[ModerationResult]


# ============================================================================
# Image Models
# ============================================================================

class ImageGenerationRequest(BaseModel):
    """Image generation request"""
    prompt: str
    model: Optional[str] = "dall-e-3"
    n: Optional[int] = 1
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    style: Optional[str] = "vivid"
    user: Optional[str] = None


class ImageObject(BaseModel):
    """Image object"""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    """Image generation response"""
    created: int
    data: List[ImageObject]


# ============================================================================
# Vector Store Models
# ============================================================================

class VectorStoreRequest(BaseModel):
    """Request to add document to vector store"""
    text: str
    embedding_model: str
    metadata: Optional[Dict[str, Any]] = None


class VectorSearchRequest(BaseModel):
    """Request to search vector store"""
    query: str
    embedding_model: str
    top_k: int = 5
    threshold: float = 0.0


class VectorDocument(BaseModel):
    """Vector document response"""
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    similarity: Optional[float] = None


# ============================================================================
# Error Models
# ============================================================================

class OpenAIError(BaseModel):
    """OpenAI-compatible error response"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI error response wrapper"""
    error: OpenAIError

