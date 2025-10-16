# OpenVINO GenAI API Server

**100% OpenAI-compatible API server** running **100% locally** on Intel NPU/CPU/GPU.

- 🏠 **Fully Local** - All AI runs on your hardware, zero cloud calls
- ⚡ **NPU Accelerated** - Intel Core Ultra Neural Processing Unit
- 🔒 **Private** - Your data never leaves your machine
- 🆓 **Free** - No API costs, unlimited usage

Use the OpenAI Python SDK with **zero code changes** - just point to `localhost:8000`.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start server (uses config.json)
python npu.py

# 3. Use with OpenAI SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

That's it! 🚀

---

## Configuration

Edit `config.json`:

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "upload_dir": "uploads",
  "vector_store_dir": "vector_store",
  "models": [
    {
      "name": "qwen2.5-3b",
      "path": "models/Qwen/Qwen2.5-3B",
      "device": "NPU",
      "type": "llm"
    }
  ]
}
```

**Device Options**: `NPU` (Intel Core Ultra), `CPU` (universal), `GPU` (for heavy models)  
**Model Types**: `llm`, `vlm`, `whisper`, `tts`, `embedding`, `text2image`, `moderation`

---

## Converting Models

Download pre-converted models or convert yourself:

```bash
# Qwen 2.5-3B (optimized for NPU)
optimum-cli export openvino -m Qwen/Qwen2.5-3B \
  --weight-format int4 --sym --ratio 1.0 --group-size 128 \
  models/Qwen/Qwen2.5-3B
```

Pre-converted models: [NPU-Optimized Collection](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0)

---

## Features

### Complete OpenAI API Compatibility

✅ Chat & Completions (streaming)  
✅ Tool/Function Calling  
✅ Structured Outputs (JSON mode & schema)  
✅ Audio (TTS & STT)  
✅ Images (generation, editing, variations)  
✅ Vision/Multimodal (VLM)  
✅ Embeddings  
✅ Content Moderation  
✅ File Management  
✅ WebSocket Realtime (voice chat)  

### Extended Features

✅ RAG with document processing  
✅ Vector store for semantic search  
✅ Multi-device support (NPU/CPU/GPU)  
✅ 7 model types  
✅ CORS & authentication  

---

## API Reference

### Setup Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)
```

### 1. Models

```python
# List all models
models = client.models.list()

# Get model details
model = client.models.retrieve("qwen2.5-3b")
```

### 2. Chat Completions

**Basic:**
```python
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is OpenVINO?"}
    ],
    temperature=0.7,
    max_tokens=200
)
```

**Streaming:**
```python
stream = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**With Seed (Reproducible):**
```python
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "Random number"}],
    seed=12345,  # Same seed = same output
    temperature=0.7
)
```

**With Stop Sequences:**
```python
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "List days of week"}],
    stop=["Thursday"],  # Stop here
    max_tokens=200
)
```

### 3. Tool Calling (Function Calling)

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
        
    # Execute function and send back result
    messages.append({
        "role": "assistant",
        "tool_calls": response.choices[0].message.tool_calls
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": "22°C, sunny"  # Your function result
    })
    
    # Get final response
    final = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=messages,
        tools=tools
    )
```

### 4. Structured Outputs

**JSON Mode:**
```python
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[
        {"role": "system", "content": "You output JSON only."},
        {"role": "user", "content": "User profile for John, age 28"}
    ],
    response_format={"type": "json_object"}
)

import json
data = json.loads(response.choices[0].message.content)
```

**JSON Schema:**
```python
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "Create user profile"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"}
            },
            "required": ["name", "age"]
        }
    }
)
```

### 5. Text Completions (Legacy)

```python
response = client.completions.create(
    model="qwen2.5-3b",
    prompt="The three laws of robotics are:",
    max_tokens=100,
    temperature=0.7
)
```

### 6. Embeddings

```python
# Single text
response = client.embeddings.create(
    model="text-embedding-model",
    input="OpenVINO accelerates AI"
)
embedding = response.data[0].embedding

# Batch processing
response = client.embeddings.create(
    model="text-embedding-model",
    input=["Text 1", "Text 2", "Text 3"]
)
```

### 7. Audio

**Speech-to-Text (Whisper):**
```python
with open("audio.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="whisper-base",
        file=f,
        response_format="json"
    )
print(transcription.text)
```

**Text-to-Speech:**
```python
response = client.audio.speech.create(
    model="speecht5-tts",
    input="Hello! This is a test.",
    voice="alloy",
    response_format="mp3"
)
response.stream_to_file("output.mp3")
```

**Voice Chat Pipeline:**
```python
# STT → Chat → TTS
transcription = client.audio.transcriptions.create(model="whisper-base", file=audio_file)
response = client.chat.completions.create(model="qwen2.5-3b", messages=[{"role": "user", "content": transcription.text}])
speech = client.audio.speech.create(model="speecht5-tts", input=response.choices[0].message.content)
speech.stream_to_file("response.mp3")
```

### 8. Images

**Generate:**
```python
response = client.images.generate(
    model="stable-diffusion",
    prompt="A serene Japanese garden",
    n=1,
    size="1024x1024",
    response_format="url"  # or "b64_json"
)
print(response.data[0].url)
```

**Edit:**
```python
response = client.images.edit(
    image=open("original.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="Add sunset in background",
    n=1
)
```

**Variations:**
```python
response = client.images.create_variation(
    image=open("original.png", "rb"),
    n=3
)
```

### 9. Vision/Multimodal

```python
import base64

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="minicpm-v",  # VLM model
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }}
        ]
    }]
)
```

### 10. Content Moderation

```python
response = client.moderations.create(
    input="Text to moderate"
)

if response.results[0].flagged:
    print("Content flagged!")
```

### 11. File Management

```python
# Upload
with open("doc.pdf", "rb") as f:
    file = client.files.create(file=f, purpose="assistants")

# List
files = client.files.list()

# Download
import requests
content = requests.get(f"http://localhost:8000/v1/files/{file.id}/content").content

# Delete
client.files.delete(file.id)
```

### 12. RAG (Document Context)

```python
# Upload document
with open("document.pdf", "rb") as f:
    file = client.files.create(file=f)

# Ask questions about it
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize this document"},
            {"type": "image_url", "image_url": {"url": file.id}}
        ]
    }]
)
```

### 13. Vector Store (Semantic Search)

```python
import requests

# Add document
response = requests.post(
    "http://localhost:8000/v1/vector_store/documents",
    json={
        "text": "OpenVINO is an AI inference toolkit.",
        "embedding_model": "text-embedding-model",
        "metadata": {"source": "docs"}
    }
)

# Search
results = requests.post(
    "http://localhost:8000/v1/vector_store/search",
    json={
        "query": "What is OpenVINO?",
        "embedding_model": "text-embedding-model",
        "top_k": 5
    }
).json()["results"]

for r in results:
    print(f"Similarity: {r['similarity']:.3f} - {r['text']}")
```

### 14. WebSocket Realtime (Voice Chat)

```python
import asyncio
import websockets
import json

async def voice_chat():
    async with websockets.connect("ws://localhost:8000/v1/realtime?model=qwen2.5-3b") as ws:
        # Receive session
        event = json.loads(await ws.recv())
        print(f"Session: {event['session']['id']}")
        
        # Send text
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello!"}]
            }
        }))
        
        # Receive response
        while True:
            event = json.loads(await ws.recv())
            if event["type"] == "response.text.delta":
                print(event["delta"], end="", flush=True)
            elif event["type"] == "response.done":
                break

asyncio.run(voice_chat())
```

---

## All API Endpoints

### Standard OpenAI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List all models |
| `/v1/models/{model}` | GET | Get model details |
| `/v1/chat/completions` | POST | Chat with streaming, tools, JSON mode |
| `/v1/completions` | POST | Text completion with streaming |
| `/v1/embeddings` | POST | Generate text embeddings |
| `/v1/audio/transcriptions` | POST | Speech-to-text (Whisper) |
| `/v1/audio/speech` | POST | Text-to-speech |
| `/v1/images/generations` | POST | Generate images |
| `/v1/images/edits` | POST | Edit images with mask |
| `/v1/images/variations` | POST | Create image variations |
| `/v1/moderations` | POST | Content moderation |
| `/v1/files` | POST | Upload file |
| `/v1/files` | GET | List files |
| `/v1/files/{id}` | GET | Get file metadata |
| `/v1/files/{id}/content` | GET | Download file |
| `/v1/files/{id}` | DELETE | Delete file |
| `/v1/realtime` | WebSocket | Real-time voice+text chat |

### Extended Endpoints (Non-OpenAI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/vector_store/documents` | POST | Add document with embedding |
| `/v1/vector_store/search` | POST | Semantic similarity search |
| `/v1/vector_store/documents` | GET | List all documents |
| `/v1/vector_store/documents/{id}` | GET | Get document |
| `/v1/vector_store/documents/{id}` | DELETE | Delete document |
| `/v1/vector_store/clear` | POST | Clear all documents |
| `/health` | GET | Server health status |

---

## Testing

Run comprehensive tests covering all 25 features:

```bash
python examples/comprehensive_test.py
```

Tests include:
- Core: Models, chat, streaming, completions
- Advanced: Tool calling, JSON mode, structured outputs
- Files: Upload, RAG, vector store
- Audio: TTS, STT, voice chat (REST + WebSocket)
- Images: Generation, editing, variations
- Vision: Multimodal support
- Safety: Content moderation
- Parameters: Seed, stop sequences, system fingerprint

---

## Prerequisites

### Intel NPU Driver (for NPU acceleration)

**Download**: [Intel NPU Driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)

**Supported Processors**: Intel® Core™ Ultra (Series 1 & 2)

**Verify NPU:**
```python
import openvino as ov
print(ov.Core().available_devices())  # Should show 'NPU'
```

---

## Project Structure

```
npu/
├── npu.py                       # Entry point
├── config.json                  # Configuration
├── requirements.txt             # Dependencies
├── app/                         # Modular application
│   ├── main.py                  # FastAPI app
│   ├── models.py                # Pydantic schemas
│   ├── managers.py              # Model/file/vector managers
│   ├── utils.py                 # Helper functions
│   ├── realtime.py              # WebSocket voice chat
│   └── routes/                  # API endpoints (9 modules)
│       ├── models.py
│       ├── chat.py
│       ├── completions.py
│       ├── audio.py
│       ├── images.py
│       ├── files.py
│       ├── embeddings.py
│       ├── moderation.py
│       └── vector_store.py
├── examples/
│   └── comprehensive_test.py    # All 25 automated tests
├── models/                      # AI models (local storage)
├── uploads/                     # Uploaded files (local)
├── vector_store/                # Vector embeddings (local)
└── generated_images/            # Generated images (local)
```

---

## Authentication (Optional)

Enable API key validation:

```bash
export OPENAI_API_KEY_REQUIRED=true
export OPENAI_API_KEYS=sk-key1,sk-key2
```

Use with client:
```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-key1"
)
```

---

## Privacy & Local Operation

### 100% Local - No Cloud

Everything runs on your hardware:
- ✅ All AI models on NPU/CPU/GPU
- ✅ All data stored locally
- ✅ No external API calls
- ✅ Works fully offline (after model download)

The **only** optional network usage:
- Model download (one-time setup)
- Remote image URLs (use base64/local files instead for 100% offline)

**Your data NEVER leaves your machine!** 🔒

---

## Performance

| Feature | Device | Latency |
|---------|--------|---------|
| Chat (Qwen 3B) | NPU | ~1-2s |
| Streaming | NPU | Real-time |
| Voice (REST) | NPU+CPU | ~3-5s |
| Voice (WebSocket) | NPU+CPU | ~200ms |
| Embeddings | CPU | ~100ms |
| Image Gen | GPU/CPU | ~5-10s |

---

## Advanced Examples

### Complete Tool Execution Flow

See API section above (#3) for full example.

### Multimodal with Multiple Images

```python
response = client.chat.completions.create(
    model="minicpm-v",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these images"},
            {"type": "image_url", "image_url": {"url": "file://image1.jpg"}},
            {"type": "image_url", "image_url": {"url": "file://image2.jpg"}}
        ]
    }]
)
```

### RAG with Vector Search

```python
import requests

# 1. Add documents to vector store
docs = [
    "OpenVINO optimizes AI models",
    "NPU accelerates inference",
    "Qwen is a language model"
]

for doc in docs:
    requests.post("http://localhost:8000/v1/vector_store/documents", json={
        "text": doc,
        "embedding_model": "text-embedding-model"
    })

# 2. Search for relevant context
results = requests.post("http://localhost:8000/v1/vector_store/search", json={
    "query": "How to speed up AI?",
    "embedding_model": "text-embedding-model",
    "top_k": 2
}).json()["results"]

# 3. Use in chat
context = "\n".join([r["text"] for r in results])
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[
        {"role": "system", "content": f"Context: {context}"},
        {"role": "user", "content": "How can I speed up my AI models?"}
    ]
)
```

---

## Resources

- **OpenAI API Spec**: https://platform.openai.com/docs/api-reference
- **OpenVINO GenAI**: https://github.com/openvinotoolkit/openvino.genai
- **Intel NPU Driver**: https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
- **Pre-converted Models**: https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0

---

## Troubleshooting

### NPU Not Detected
- Install Intel NPU Driver (link above)
- Verify: Intel Core Ultra Series 1 or 2 processor
- Check: `ov.Core().available_devices()` shows 'NPU'

### Server Won't Start
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Verify OpenVINO installation
python -c "import openvino as ov; print(ov.get_version())"
```

### Import Errors
All dependencies are in `requirements.txt` - just run:
```bash
pip install -r requirements.txt
```

---

## License

Apache 2.0
