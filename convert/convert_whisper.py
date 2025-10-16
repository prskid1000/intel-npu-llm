#!/usr/bin/env python3
"""
OpenVINO Whisper Model Conversion Script
Converts Whisper-base for Speech-to-Text (STT) functionality
"""

import sys
from pathlib import Path
import warnings

def convert_whisper():
    """Convert Whisper-base speech recognition model to OpenVINO"""
    
    print("=" * 60)
    print("Whisper Model Conversion (whisper-base)")
    print("=" * 60)
    print()
    
    model_id = "openai/whisper-base"
    out_dir = Path("models/Whisper/whisper-base")
    
    print(f"Source Model: {model_id}")
    print(f"Output Directory: {out_dir}")
    print(f"Task: Automatic speech recognition (ASR)")
    print()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor
        
        print("✓ Required packages loaded")
        print()
        
    except ImportError as e:
        print(f"❌ ERROR: Missing required package: {e}")
        print()
        print("Please ensure you have installed:")
        print("  pip install optimum-intel[openvino]")
        return False
    
    # Check if already converted
    if (out_dir / "openvino_encoder_model.xml").exists():
        print("✓ Model already converted (found in output directory)")
        print()
        return True
    
    print("Converting model...")
    print("This may take a few minutes...")
    print()
    
    try:
        # Convert model
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )
        
        # Save processor (tokenizer + feature extractor)
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Save both
        model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)
        
        print("✓ Model converted successfully")
        print()
        
    except Exception as e:
        print(f"❌ ERROR during conversion: {e}")
        return False
    
    # Success!
    print("=" * 60)
    print("✅ Whisper Model Conversion Complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {out_dir}")
    print()
    
    return True


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    success = convert_whisper()
    
    if not success:
        print()
        print("Conversion failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

