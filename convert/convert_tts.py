#!/usr/bin/env python3
"""
OpenVINO TTS Model Conversion Script
Converts SpeechT5 for Text-to-Speech (TTS) functionality
"""

import sys
from pathlib import Path
import warnings

def convert_tts():
    """Convert SpeechT5 TTS model to OpenVINO"""
    
    print("=" * 60)
    print("TTS Model Conversion (SpeechT5)")
    print("=" * 60)
    print()
    
    model_id = "microsoft/speecht5_tts"
    out_dir = Path("models/TTS/speecht5-tts")
    
    print(f"Source Model: {model_id}")
    print(f"Output Directory: {out_dir}")
    print(f"Task: Text-to-speech synthesis")
    print()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.intel.openvino import OVModelForTextToWaveform
        from transformers import AutoProcessor
        
        print("✓ Required packages loaded")
        print()
        
    except ImportError:
        # Fallback to optimum-cli command
        print("✓ Using optimum-cli for conversion")
        print()
        
        # Check if already converted
        if (out_dir / "openvino_encoder_model.xml").exists():
            print("✓ Model already converted (found in output directory)")
            print()
            return True
        
        print("Converting model...")
        print("This may take a few minutes...")
        print()
        
        try:
            import subprocess
            
            # Use optimum-cli to export with vocoder
            cmd = [
                "optimum-cli", "export", "openvino",
                "--model", model_id,
                "--task", "text-to-audio",
                "--model-kwargs", '{"vocoder": "microsoft/speecht5_hifigan"}',
                str(out_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ ERROR during conversion:")
                print(result.stderr)
                return False
            
            print("✓ Model converted successfully")
            print()
            
        except Exception as e:
            print(f"❌ ERROR during conversion: {e}")
            return False
        
        # Success!
        print("=" * 60)
        print("✅ TTS Model Conversion Complete!")
        print("=" * 60)
        print()
        print(f"Model saved to: {out_dir}")
        print()
        
        return True
    
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
        model = OVModelForTextToWaveform.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )
        
        # Save processor (includes tokenizer)
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
    print("✅ TTS Model Conversion Complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {out_dir}")
    print()
    
    return True


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    success = convert_tts()
    
    if not success:
        print()
        print("Conversion failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

