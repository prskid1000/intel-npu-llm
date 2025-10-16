#!/usr/bin/env python3
"""
OpenVINO Text2Image Model Conversion Script
Converts Stable Diffusion 1.5 for image generation
"""

import sys
from pathlib import Path
import warnings

def convert_text2image():
    """Convert Stable Diffusion 1.5 model to OpenVINO"""
    
    print("=" * 60)
    print("Text2Image Model Conversion (Stable Diffusion 1.5)")
    print("=" * 60)
    print()
    
    model_id = "runwayml/stable-diffusion-v1-5"
    out_dir = Path("models/Text2Image/sd-1.5")
    
    print(f"Source Model: {model_id}")
    print(f"Output Directory: {out_dir}")
    print(f"Task: Text-to-image generation")
    print(f"Format: FP16 (CPU-optimized)")
    print()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.intel import OVStableDiffusionPipeline
        
        print("✓ Required packages loaded")
        print()
        
    except ImportError as e:
        print(f"❌ ERROR: Missing required package: {e}")
        print()
        print("Please ensure you have installed:")
        print("  pip install optimum-intel[openvino,diffusers]")
        return False
    
    # Check if already converted
    if (out_dir / "vae_decoder" / "openvino_model.xml").exists():
        print("✓ Model already converted (found in output directory)")
        print()
        return True
    
    print("Converting Stable Diffusion pipeline...")
    print("This includes: UNet, VAE, Text Encoder, Safety Checker")
    print("This may take 5-10 minutes...")
    print()
    
    try:
        # Convert entire pipeline
        pipeline = OVStableDiffusionPipeline.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )
        
        # Save pipeline
        pipeline.save_pretrained(out_dir)
        
        print("✓ Stable Diffusion pipeline converted successfully")
        print()
        
    except Exception as e:
        print(f"❌ ERROR during conversion: {e}")
        return False
    
    # Success!
    print("=" * 60)
    print("✅ Text2Image Model Conversion Complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {out_dir}")
    print()
    print("Note: SD 1.5 works on CPU without GPU!")
    print()
    
    return True


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    success = convert_text2image()
    
    if not success:
        print()
        print("Conversion failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

