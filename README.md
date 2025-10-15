# OpenVINO GenAI API Server

A comprehensive OpenAI-compatible API server for running AI models on Intel NPU, CPU, and GPU using OpenVINO. Supports text generation, vision, audio processing, and RAG (Retrieval Augmented Generation).

## Prerequisites

### Intel NPU Driver (Windows)

Before running models on NPU, ensure you have the latest Intel NPU Driver installed:

📥 [Download Intel NPU Driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)

**Supported Processors:**
- Intel® Core™ Ultra processors (Series 1 and Series 2)
- Includes H, U, HX, V, and Desktop variants

**Latest Version:** 32.0.100.4297

## Setup

### 1. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Verify NPU Availability

Check if your NPU is properly detected:

```python
import openvino as ov
core = ov.Core()
devices = core.available_devices()
print("Available devices:", devices)
# Should show 'NPU' in the list if driver is installed correctly
```

## Model Conversion

Convert models from HuggingFace to OpenVINO format for optimal performance.

### Text Models (LLM)

**Qwen 2.5-3B**
```bash
optimum-cli export openvino -m Qwen/Qwen2.5-3B \
  --weight-format int4 --sym --ratio 1.0 --group-size 128 \
  models/Qwen/Qwen2.5-3B
```

**Other LLMs**
```bash
# Llama models
optimum-cli export openvino -m meta-llama/Llama-3.2-3B \
  --weight-format int4 \
  models/Llama-3.2-3B

# Phi models
optimum-cli export openvino -m microsoft/Phi-3-mini-4k-instruct \
  --weight-format int4 \
  models/Phi-3-mini
```

### Vision Models (VLM)

**MiniCPM-V**
```bash
pip install timm einops

optimum-cli export openvino -m openbmb/MiniCPM-V-2_6 \
  --trust-remote-code --weight-format int4 \
  models/MiniCPM-V-2_6
```

**InternVL2**
```bash
pip install timm einops

optimum-cli export openvino -m OpenGVLab/InternVL2-1B \
  --trust-remote-code --weight-format int4 \
  models/InternVL2-1B
```

### Audio Models

**Whisper (Speech-to-Text)**
```bash
# Whisper base
optimum-cli export openvino --model openai/whisper-base \
  models/whisper-base

# Whisper base with quantization
optimum-cli export openvino --model openai/whisper-base \
  --disable-stateful --quant-mode int8 \
  --dataset librispeech --num-samples 32 \
  models/whisper-base-int8
```

**SpeechT5 (Text-to-Speech)**
```bash
optimum-cli export openvino --model microsoft/speecht5_tts \
  --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" \
  models/speecht5_tts
```

## Pre-converted Models

You can also use pre-optimized models from the OpenVINO collection:

🔗 [LLMs Optimized for NPU](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0)

## Running Converted Models

You can run inference in two ways:
1. **Python API** - Direct inference using OpenVINO GenAI
2. **OpenAI-Compatible Server** - Production-ready server with NPU support (⭐ **Recommended**)

## Option 1: Python API (OpenVINO GenAI)

Use [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) for direct Python inference.

### Basic Text Generation

```python
import openvino_genai as ov_genai

# Initialize the pipeline with your converted model
pipe = ov_genai.LLMPipeline("./Qwen/Qwen2.5-3B", "NPU")

# Generate text
result = pipe.generate("What is OpenVINO?", max_new_tokens=100)
print(result)
```

### Streaming Generation

Stream tokens as they're generated for real-time output:

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("./Qwen/Qwen2.5-3B", "NPU")

# Stream tokens in real-time
for token in pipe.generate("Explain neural networks", max_new_tokens=200, stream=True):
    print(token, end='', flush=True)
```

### Using Different Devices

```python
import openvino_genai as ov_genai

# NPU for AI inference (recommended for Intel Core Ultra)
pipe_npu = ov_genai.LLMPipeline("./Qwen/Qwen2.5-3B", "NPU")

# CPU fallback
pipe_cpu = ov_genai.LLMPipeline("./Qwen/Qwen2.5-3B", "CPU")

# GPU (if available)
pipe_gpu = ov_genai.LLMPipeline("./Qwen/Qwen2.5-3B", "GPU")

result = pipe_npu.generate("Hello!", max_new_tokens=50)
print(result)
```

### Advanced Configuration

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("./Qwen/Qwen2.5-3B", "NPU")

# Configure generation parameters
config = ov_genai.GenerationConfig()
config.max_new_tokens = 100
config.temperature = 0.7
config.top_p = 0.9
config.do_sample = True

result = pipe.generate("Tell me a story", config)
print(result)
```

## Option 2: OpenAI-Compatible API Server (⭐ Recommended)

A production-ready FastAPI server that provides comprehensive OpenAI-compatible endpoints with full NPU support.

### Features

#### 🎯 Core Capabilities
✅ **OpenAI API Compatible** - Drop-in replacement for OpenAI API  
✅ **NPU Acceleration** - Full NPU support for Intel Core Ultra processors  
✅ **Multi-Model Support** - Serve multiple models simultaneously on different devices  
✅ **Streaming** - Real-time token streaming for all text generation

#### 🤖 AI Capabilities
✅ **Text Generation** - LLM chat and completion endpoints  
✅ **Vision (Multimodal)** - Process images with vision-language models  
✅ **Speech-to-Text** - Whisper-based audio transcription  
✅ **Text-to-Speech** - Generate natural-sounding speech  
✅ **RAG Support** - File upload and document processing for context

#### 📁 File & Document Processing
✅ **File Upload API** - OpenAI-compatible file management  
✅ **Document Extraction** - Support for PDF, DOCX, TXT, JSON  
✅ **Image Processing** - Load from URLs, base64, or file uploads  
✅ **Audio Processing** - WAV, MP3, FLAC, and more  

### Quick Start

**1. Install Dependencies:**
```powershell
pip install -r requirements.txt
```

**2. Configure Models (config.json):**
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
    },
    {
      "name": "minicpm-v",
      "path": "models/MiniCPM-V-2_6",
      "device": "CPU",
      "type": "vlm"
    },
    {
      "name": "whisper-base",
      "path": "models/whisper-base",
      "device": "CPU",
      "type": "whisper"
    },
    {
      "name": "speecht5-tts",
      "path": "models/speecht5_tts",
      "device": "CPU",
      "type": "tts"
    }
  ]
}
```

**Model Types:**
- `llm` - Large Language Models (text generation)
- `vlm` - Vision-Language Models (multimodal)
- `whisper` - Speech-to-text models
- `tts` - Text-to-speech models
- `embedding` - Embedding models (future support)

**3. Start Server:**
```powershell
python npu.py
```

The server will start at `http://localhost:8000`

### API Endpoints

The server provides OpenAI-compatible endpoints organized by capability:

#### 📝 Text Generation

**List Models**
```bash
curl http://localhost:8000/v1/models
```

**Chat Completions**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "messages": [
      {"role": "user", "content": "What is OpenVINO?"}
    ],
    "max_tokens": 100
  }'
```

**Streaming Chat**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

**Text Completions**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "prompt": "The future of AI is",
    "max_tokens": 50
  }'
```

#### 👁️ Vision (Multimodal)

**Vision Chat with Image URL**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm-v",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

**Vision Chat with Base64 Image**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm-v",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
        ]
      }
    ]
  }'
```

#### 📁 File Management

**Upload File**
```bash
curl http://localhost:8000/v1/files \
  -F "file=@document.pdf" \
  -F "purpose=assistants"
```

**List Files**
```bash
curl http://localhost:8000/v1/files
```

**Get File Info**
```bash
curl http://localhost:8000/v1/files/file-abc123
```

**Download File Content**
```bash
curl http://localhost:8000/v1/files/file-abc123/content
```

**Delete File**
```bash
curl -X DELETE http://localhost:8000/v1/files/file-abc123
```

#### 🔊 Audio Processing

**Speech-to-Text (Transcription)**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-base" \
  -F "response_format=json"
```

**Text-to-Speech**
```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "speecht5-tts",
    "input": "Hello, this is a test of text to speech.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

#### 🔍 RAG (Retrieval Augmented Generation)

**Chat with Uploaded Document**
```bash
# First, upload a document
FILE_ID=$(curl http://localhost:8000/v1/files \
  -F "file=@document.pdf" \
  -F "purpose=assistants" | jq -r '.id')

# Then reference it in chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Summarize this document"},
          {"type": "image_url", "image_url": {"url": "'$FILE_ID'"}}
        ]
      }
    ]
  }'
```

### Client Examples

#### Python (OpenAI Library)

**Basic Text Chat**
```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # API key not required but library needs it
)

# Chat completion
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=200
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Vision (Multimodal)**
```python
import base64

# Encode image to base64
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Vision chat
response = client.chat.completions.create(
    model="minicpm-v",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

**File Upload & RAG**
```python
# Upload a document
with open("document.pdf", "rb") as f:
    file = client.files.create(file=f, purpose="assistants")

print(f"Uploaded: {file.id}")

# Use document in chat for RAG
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this document"},
                {"type": "image_url", "image_url": {"url": file.id}}
            ]
        }
    ]
)
print(response.choices[0].message.content)

# List all files
files = client.files.list()
for f in files.data:
    print(f"{f.filename}: {f.id}")

# Delete file
client.files.delete(file.id)
```

**Audio - Speech-to-Text**
```python
# Transcribe audio
with open("audio.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="whisper-base",
        file=f,
        response_format="json"
    )

print(transcription.text)
```

**Audio - Text-to-Speech**
```python
# Generate speech
response = client.audio.speech.create(
    model="speecht5-tts",
    input="Hello! This is a test of text to speech.",
    voice="alloy",
    response_format="mp3"
)

# Save to file
response.stream_to_file("output.mp3")
```

**Complete Multimodal Pipeline**
```python
# 1. Generate voice note from text
voice_response = client.audio.speech.create(
    model="speecht5-tts",
    input="What is machine learning?",
    response_format="wav"
)
voice_response.stream_to_file("question.wav")

# 2. Transcribe voice note
with open("question.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="whisper-base",
        file=f
    )

# 3. Upload an image for context
with open("ml_diagram.png", "rb") as f:
    img_file = client.files.create(file=f, purpose="vision")

# 4. Ask question with image context
response = client.chat.completions.create(
    model="minicpm-v",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": transcription.text},
                {"type": "image_url", "image_url": {"url": img_file.id}}
            ]
        }
    ]
)

# 5. Convert answer to speech
answer_audio = client.audio.speech.create(
    model="speecht5-tts",
    input=response.choices[0].message.content,
    response_format="mp3"
)
answer_audio.stream_to_file("answer.mp3")
```

#### JavaScript/TypeScript
```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy'
});

const response = await client.chat.completions.create({
  model: 'qwen2.5-3b',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});

console.log(response.choices[0].message.content);
```

#### cURL
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Multi-Model Configuration

Serve multiple models on different devices:

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "models": [
    {
      "name": "qwen2.5-3b-npu",
      "path": "models/Qwen/Qwen2.5-3B/1",
      "device": "NPU"
    },
    {
      "name": "qwen2.5-7b-cpu",
      "path": "models/Qwen/Qwen2.5-7B/1",
      "device": "CPU"
    },
    {
      "name": "llama-3.2-gpu",
      "path": "models/Llama-3.2-3B/1",
      "device": "GPU"
    }
  ]
}
```

Then use different models in your requests:
```python
# Use NPU model
response = client.chat.completions.create(
    model="qwen2.5-3b-npu",
    messages=[...]
)

# Use CPU model
response = client.chat.completions.create(
    model="qwen2.5-7b-cpu",
    messages=[...]
)
```

### Production Deployment

#### Using Uvicorn Directly
```powershell
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

#### Docker (Future)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "server.py"]
```

### Advanced Configuration

#### Generation Parameters
```python
response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    temperature=0.8,      # Creativity (0.0-2.0)
    top_p=0.95,          # Nucleus sampling
    max_tokens=500,      # Maximum response length
    presence_penalty=0,  # Penalize repeated topics
    frequency_penalty=0  # Penalize repeated words
)
```

### Example Scripts

The repository includes ready-to-use example scripts:

**`npu_test.py`** - Basic OpenAI client examples
```bash
python npu_test.py
```

**`multimodal_example.py`** - File upload, RAG, and multimodal examples
```bash
python multimodal_example.py
```

**`audio_example.py`** - Speech-to-text and text-to-speech examples
```bash
python audio_example.py
```

## Supported File Formats

### Documents (RAG)
- **Text**: `.txt`, `.json`
- **PDF**: `.pdf` (requires `PyPDF2`)
- **Word**: `.doc`, `.docx` (requires `python-docx`)

### Images (Vision)
- **Formats**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`
- **Input**: URLs, base64 data URI, file uploads, or local paths

### Audio
- **Input**: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`
- **Output**: `.wav`, `.mp3`, `.flac`, `.aac`, `.opus`, `.pcm`

## Model Storage Locations

### HuggingFace Cache (Downloaded Models)
- **Windows**: `C:\Users\<YourUsername>\.cache\huggingface\hub\`
- **Custom cache**: Set `HF_HOME` environment variable

### Converted OpenVINO Models
Models are saved to the output directory specified in the conversion command (e.g., `./Qwen/Qwen2.5-3B`)

To check your cache:
```bash
huggingface-cli scan-cache
```

## Troubleshooting

### NPU Not Detected
- Ensure Intel NPU Driver is installed: [Download Link](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)
- Verify processor has NPU (Intel Core Ultra Series 1 or 2)
- Check device availability:
  ```python
  import openvino as ov
  print(ov.Core().available_devices())
  ```

### Audio Processing Issues
- Install audio dependencies:
  ```bash
  pip install librosa soundfile pydub
  ```
- For format conversion, ensure FFmpeg is installed

### Vision Model Issues
- Install vision dependencies:
  ```bash
  pip install timm einops
  ```
- Use `--trust-remote-code` when converting VLMs

### File Upload Issues
- Check `upload_dir` permissions in `config.json`
- Ensure sufficient disk space
- For large files, consider increasing server timeout

## Resources

- [OpenVINO GenAI Repository](https://github.com/openvinotoolkit/openvino.genai)
- [OpenVINO GenAI Documentation](https://openvinotoolkit.github.io/openvino.genai/)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum/intel/index)
- [List of Supported Models](https://github.com/openvinotoolkit/openvino.genai#supported-models)
- [OpenVINO Whisper Examples](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python)
- [Intel NPU Documentation](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_NPU.html)

## License

This project uses OpenVINO GenAI which is licensed under Apache 2.0.