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
        
        # Add detokenizer for OpenVINO GenAI
        try:
            print("Converting tokenizer and detokenizer to OpenVINO format...")
            from transformers import AutoTokenizer
            from openvino_tokenizers import convert_tokenizer
            import openvino as ov
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Create tokenizer
            ov_tokenizer = convert_tokenizer(tokenizer)
            tokenizer_path = out_dir / "openvino_tokenizer.xml"
            ov.save_model(ov_tokenizer, str(tokenizer_path))
            print("✓ OpenVINO tokenizer created")
            
            # Create detokenizer - convert_tokenizer with with_detokenizer=True returns (tokenizer, detokenizer)
            ov_tokenizer_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True, skip_special_tokens=True)
            detokenizer_path = out_dir / "openvino_detokenizer.xml"
            ov.save_model(ov_tokenizer_detokenizer[1], str(detokenizer_path))
            print("✓ OpenVINO detokenizer created")
            print()
        except Exception as e:
            print(f"⚠️  Warning: Could not create detokenizer: {e}")
            print("   Model may work without it for some use cases")
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

