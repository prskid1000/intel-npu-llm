#!/bin/bash
# ============================================================================
# OpenVINO GenAI - Complete Model Compilation Script
# Compiles all models needed for full API functionality
# Uses Phi-3.5-vision-instruct as unified LLM+VLM model
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "OpenVINO GenAI Model Compilation"
echo "=========================================="
echo ""

# Create model directories
mkdir -p models/{Phi,Embeddings,Whisper,TTS,Text2Image,Moderation}

# ============================================================================
# 1. Phi-3.5-Vision - Unified LLM + VLM (REQUIRED)
# ============================================================================
echo "📝 [1/6] Compiling Phi-3.5-Vision (Multimodal LLM+VLM)..."
echo "   This replaces both Qwen (LLM) and MiniCPM-V (VLM)"
echo "   Features: Text chat, vision, 4B params, 128K context"
optimum-cli export openvino \
  --model microsoft/Phi-3.5-vision-instruct \
  --weight-format int4 \
  --trust-remote-code \
  models/Phi/Phi-3.5-vision-instruct
echo "✅ Phi-3.5-vision-instruct compiled"
echo ""

# ============================================================================
# 2. Embedding Model (REQUIRED for RAG/Vector Store)
# ============================================================================
echo "🔍 [2/6] Compiling Embedding Model - BGE-small-en-v1.5..."
optimum-cli export openvino \
  --model BAAI/bge-small-en-v1.5 \
  --task feature-extraction \
  models/Embeddings/bge-small-en
echo "✅ BGE-small-en-v1.5 compiled"
echo ""

# ============================================================================
# 3. Whisper - Speech-to-Text
# ============================================================================
echo "🎤 [3/6] Compiling Whisper - whisper-base..."
optimum-cli export openvino \
  --model openai/whisper-base \
  --task automatic-speech-recognition \
  models/Whisper/whisper-base
echo "✅ Whisper-base compiled"
echo ""

# ============================================================================
# 4. TTS - Text-to-Speech
# ============================================================================
echo "🔊 [4/6] Compiling TTS - SpeechT5..."
optimum-cli export openvino \
  --model microsoft/speecht5_tts \
  --task text-to-audio \
  models/TTS/speecht5-tts
echo "✅ SpeechT5 TTS compiled"
echo ""

# ============================================================================
# 5. Text2Image - Image Generation
# ============================================================================
echo "🎨 [5/6] Compiling Text2Image - Stable Diffusion 1.5..."
echo "   SD 1.5: CPU inference - no GPU needed!"
echo "   Using fp16 for quality (slower on CPU but works on any system)"
optimum-cli export openvino \
  --model runwayml/stable-diffusion-v1-5 \
  --task stable-diffusion \
  --weight-format fp16 \
  models/Text2Image/sd-1.5
echo "✅ Stable Diffusion 1.5 compiled"
echo ""

# ============================================================================
# 6. Moderation - Content Safety
# ============================================================================
echo "🛡️ [6/6] Compiling Moderation - Toxic-BERT..."
optimum-cli export openvino \
  --model unitary/toxic-bert \
  --task text-classification \
  models/Moderation/toxic-bert
echo "✅ Toxic-BERT compiled"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "✅ All Models Compiled Successfully!"
echo "=========================================="
echo ""
echo "Models installed (CPU/NPU Only - No GPU Required!):"
echo "  ✓ LLM+VLM: Phi-3.5-vision-instruct (NPU) - 4B params, 128K context"
echo "    - Handles both text chat AND vision/multimodal tasks"
echo "    - Replaces Qwen (LLM) + MiniCPM-V (VLM)"
echo "  ✓ Embedding: BGE-small-en-v1.5 (CPU)"
echo "  ✓ Whisper: whisper-base (CPU)"
echo "  ✓ TTS: SpeechT5 (CPU)"
echo "  ✓ Text2Image: Stable Diffusion 1.5 (CPU) - No GPU needed!"
echo "  ✓ Moderation: Toxic-BERT (CPU)"
echo ""
echo "Total RAM requirements:"
echo "  • System RAM: 12-14GB (all models on CPU/NPU)"
echo "  • GPU VRAM: 0GB - No dedicated GPU required!"
echo "  • Recommended: 16GB+ RAM"
echo ""
echo "Device allocation:"
echo "  • NPU: Phi-3.5-vision (4-6GB)"
echo "  • CPU: All other models (6-8GB total)"
echo "  • Works on: Intel Core Ultra with integrated graphics!"
echo ""
echo "Next steps:"
echo "  1. Copy config_full.json to config.json"
echo "  2. Run: python npu.py"
echo "  3. Test with: python examples/test.py"
echo ""

