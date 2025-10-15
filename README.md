# OpenVINO NPU Model Optimization

This project provides tools for converting and optimizing Large Language Models (LLMs) for Intel NPU using OpenVINO.

## Setup

### 1. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

## Model Conversion

Use any of the following commands to convert models to OpenVINO format with INT4 quantization:

### Qwen 2.5-3B

```bash
optimum-cli export openvino -m Qwen/Qwen2.5-3B --weight-format int4 --sym --ratio 1.0 --group-size 128 Qwen/Qwen2.5-3B
```

### Llama 2 7B Chat

```bash
optimum-cli export openvino -m meta-llama/Llama-2-7b-chat-hf --weight-format int4 --sym --ratio 1.0 --group-size -1 Llama-2-7b-chat-hf
```

### Llama 2 7B Chat (GPTQ)

```bash
optimum-cli export openvino -m TheBloke/Llama-2-7B-Chat-GPTQ
```

## Pre-converted Models

You can also use pre-optimized models from the OpenVINO collection:

🔗 [LLMs Optimized for NPU](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu-686e7f0bf7bc184bd71f8ba0)

## Resources

- [OpenVINO GenAI Documentation](https://github.com/openvinotoolkit/openvino.genai)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum/intel/index)