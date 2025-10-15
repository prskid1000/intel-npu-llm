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
2. **Inference Server** - Production-ready server using OpenVINO Model Server (OVMS)

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

## Option 2: Inference Server (OpenVINO Model Server)

For production deployments, use OpenVINO Model Server (OVMS) to serve your models via REST/gRPC APIs.

### Setup Environment

First, set up the OVMS environment which will activate your virtual environment and add OVMS to your PATH:

**PowerShell** (use dot-sourcing to run in current session):
```powershell
. .\ovms\setupvars.ps1
```

**Command Prompt:**
```batch
.\ovms\setupvars.bat
```

> **Note:** In PowerShell, the dot (`.`) before the script path is important - it ensures the script runs in your current session so environment changes persist.

This script will:
- ✅ Activate your `.venv` virtual environment
- ✅ Add OVMS binaries and DLLs to your PATH
- ✅ Configure the environment for running the model server

### Start the Model Server

After running the setup script, you can start the server:

```powershell
ovms --model_path models/Qwen/Qwen2.5-3B --model_name qwen --port 9000
```

### Server Options

```powershell
# Specify NPU device
ovms --model_path models/Qwen/Qwen2.5-3B --model_name qwen --port 9000 --target_device NPU

# Enable logging
ovms --model_path models/Qwen/Qwen2.5-3B --model_name qwen --port 9000 --log_level DEBUG

# Serve multiple models with config file
ovms --config_path config.json --port 9000
```

### Client Usage

Once the server is running, you can make inference requests:

**Python Client:**
```python
import requests

url = "http://localhost:9000/v2/models/qwen/infer"
payload = {
    "inputs": [{"name": "input", "data": ["What is OpenVINO?"]}],
    "parameters": {"max_new_tokens": 100}
}

response = requests.post(url, json=payload)
print(response.json())
```

**cURL:**
```bash
curl -X POST http://localhost:9000/v2/models/qwen/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "data": ["Hello!"]}]}'
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