# OpenVINO NPU Model Optimization

This project provides tools for converting and optimizing Large Language Models (LLMs) for Intel NPU using OpenVINO.

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

Use any of the following commands to convert models to OpenVINO format with INT4 quantization:

### Qwen 2.5-3B

```bash
optimum-cli export openvino -m Qwen/Qwen2.5-3B --weight-format int4 --sym --ratio 1.0 --group-size 128 models/Qwen/Qwen2.5-3B
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

## Option 2: OpenAI-Compatible API Server (⭐ Recommended for NPU)

A production-ready FastAPI server that provides OpenAI-compatible endpoints with full NPU support.

### Features

✅ **OpenAI API Compatible** - Drop-in replacement for OpenAI API  
✅ **NPU Acceleration** - Full NPU support (unlike OVMS)  
✅ **Multi-Model Support** - Serve multiple models simultaneously  
✅ **Streaming** - Real-time token streaming  
✅ **Easy Integration** - Works with OpenAI client libraries  

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
  "models": [
    {
      "name": "qwen2.5-3b",
      "path": "models/Qwen/Qwen2.5-3B/1",
      "device": "NPU"
    }
  ]
}
```

**3. Start Server:**
```powershell
python server.py
```

The server will start at `http://localhost:8000`

### API Endpoints

#### List Models
```bash
curl http://localhost:8000/v1/models
```

#### Chat Completions
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

#### Streaming Chat
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

#### Text Completions
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b",
    "prompt": "The future of AI is",
    "max_tokens": 50
  }'
```

### Client Examples

#### Python (OpenAI Library)
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

## Resources

- [OpenVINO GenAI Repository](https://github.com/openvinotoolkit/openvino.genai)
- [OpenVINO GenAI Documentation](https://openvinotoolkit.github.io/openvino.genai/)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum/intel/index)
- [List of Supported Models](https://github.com/openvinotoolkit/openvino.genai#supported-models)