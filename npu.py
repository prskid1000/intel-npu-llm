"""
OpenVINO GenAI API Server - Entry Point

100% OpenAI-compatible API server running locally on Intel NPU/CPU/GPU

Features:
- Text chat and completion (LLM) with streaming
- Vision/multimodal support (VLM)
- Audio transcription (Whisper) and TTS
- Text embeddings and vector store
- Content moderation
- Image generation, editing, variations
- Tool/function calling
- Structured outputs (JSON mode & schema)
- File uploads and RAG
- WebSocket realtime voice chat

Run: python npu.py
Docs: http://localhost:8000/docs
Test: python examples/comprehensive_test.py
"""

import uvicorn
from app.main import app, config

def main():
    """Run the server"""
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=False
    )


if __name__ == "__main__":
    main()
