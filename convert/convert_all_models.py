#!/usr/bin/env python3
"""
OpenVINO GenAI - Complete Model Compilation Script
Orchestrates conversion of all models needed for full API functionality
Uses Phi-3.5-vision-instruct as unified LLM+VLM model
"""

import sys
import time
from pathlib import Path

# Import individual converters
from convert_phi3_vision import convert_phi3_vision
from convert_deepseek_vl import convert_deepseek_vl
from convert_embeddings import convert_embeddings
from convert_whisper import convert_whisper
from convert_tts import convert_tts
from convert_text2image import convert_text2image
from convert_ssd1b import convert_ssd1b
from convert_moderation import convert_moderation


def print_header():
    """Print script header"""
    print()
    print("=" * 70)
    print("OpenVINO GenAI - Complete Model Compilation")
    print("=" * 70)
    print()
    print("This will convert all models needed for full OpenAI API compatibility:")
    print("  1. Phi-3.5-Vision (LLM + VLM) - Text chat & vision")
    print("  3. BGE-small-en-v1.5 (Embeddings) - RAG & vector search")
    print("  4. Whisper-base (STT) - Speech-to-text")
    print("  5. SpeechT5 (TTS) - Text-to-speech")
    print("  6. Stable Diffusion 1.5 (Text2Image) - Image generation")
    print("  7. SSD-1B (Text2Image) - Fast image generation")
    print("  8. Toxic-BERT (Moderation) - Content safety")
    print()
    print("Device Support: CPU/NPU - No GPU required!")
    print()
    print("=" * 70)
    print()


def print_summary(results):
    """Print conversion summary"""
    print()
    print("=" * 70)
    print("Conversion Summary")
    print("=" * 70)
    print()
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status} - {model}")
    
    print()
    
    all_success = all(results.values())
    
    if all_success:
        print("=" * 70)
        print("🎉 All Models Compiled Successfully!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Copy config_full.json to config.json")
        print("  2. Run: python npu.py")
        print("  3. Test with: python examples/test.py")
        print()
        print("Device allocation:")
        print("  - NPU: Phi-3.5-vision (4-6GB)")
        print("  - CPU: All other models (6-8GB total)")
        print()
        print("Total RAM requirements: 12-14GB recommended")
        print()
    else:
        failed = [m for m, s in results.items() if not s]
        print("=" * 70)
        print("⚠️  Some Models Failed to Convert")
        print("=" * 70)
        print()
        print(f"Failed models: {', '.join(failed)}")
        print()
        print("You can:")
        print("  - Retry individual conversions by running their scripts")
        print("  - Check error messages above for troubleshooting")
        print("  - Continue with successfully converted models")
        print()
    
    return all_success


def main():
    """Main orchestration function"""
    print_header()
    
    # Track results
    results = {}
    start_time = time.time()
    
    # Model conversion sequence
    conversions = [
        ("Phi-3.5-Vision (LLM+VLM)", convert_phi3_vision),
        ("Embeddings (BGE-small-en-v1.5)", convert_embeddings),
        ("Whisper (Speech-to-Text)", convert_whisper),
        ("TTS (SpeechT5)", convert_tts),
        ("Text2Image (Stable Diffusion 1.5)", convert_text2image),
        ("Text2Image (SSD-1B)", convert_ssd1b),
        ("Moderation (Toxic-BERT)", convert_moderation),
    ]
    
    # Run conversions
    for idx, (name, convert_func) in enumerate(conversions, 1):
        print(f"\n[{idx}/{len(conversions)}] Converting {name}...\n")
        print("-" * 70)
        print()
        
        try:
            success = convert_func()
            results[name] = success
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR in {name}: {e}\n")
            results[name] = False
        
        print()
        print("-" * 70)
        
        # Brief pause between conversions
        if idx < len(conversions):
            time.sleep(1)
    
    # Calculate total time
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    # Print summary
    all_success = print_summary(results)
    
    print(f"Total conversion time: {minutes}m {seconds}s")
    print()
    
    return 0 if all_success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion interrupted by user")
        print()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}\n")
        sys.exit(1)

