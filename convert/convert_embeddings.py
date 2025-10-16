#!/usr/bin/env python3
"""
OpenVINO Embeddings Model Conversion Script
Converts BGE-small-en-v1.5 for RAG/Vector Store functionality
"""

import sys
from pathlib import Path
import warnings

def convert_embeddings():
    """Convert BGE-small-en-v1.5 embedding model to OpenVINO"""
    
    print("=" * 60)
    print("Embeddings Model Conversion (BGE-small-en-v1.5)")
    print("=" * 60)
    print()
    
    model_id = "BAAI/bge-small-en-v1.5"
    out_dir = Path("models/Embeddings/bge-small-en")
    
    print(f"Source Model: {model_id}")
    print(f"Output Directory: {out_dir}")
    print(f"Task: Feature extraction (embeddings)")
    print()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.intel import OVModelForFeatureExtraction
        from transformers import AutoTokenizer
        
        print("✓ Required packages loaded")
        print()
        
    except ImportError as e:
        print(f"❌ ERROR: Missing required package: {e}")
        print()
        print("Please ensure you have installed:")
        print("  pip install optimum-intel[openvino]")
        return False
    
    # Check if already converted
    if (out_dir / "openvino_model.xml").exists():
        print("✓ Model already converted (found in output directory)")
        print()
        return True
    
    print("Converting model...")
    print()
    
    try:
        # Convert model with dynamic shapes for flexible batch sizes
        model = OVModelForFeatureExtraction.from_pretrained(
            model_id,
            export=True,
            compile=False,
            # Enable dynamic shapes to handle variable input sizes
            dynamic_shapes=True,
        )
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Save both
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        
        print("✓ Model converted successfully with dynamic shapes")
        print()
        
        # Add OpenVINO tokenizer for better compatibility
        try:
            print("Converting tokenizer to OpenVINO format...")
            from openvino_tokenizers import convert_tokenizer
            import openvino as ov
            
            # Create OpenVINO tokenizer
            ov_tokenizer = convert_tokenizer(tokenizer)
            tokenizer_path = out_dir / "openvino_tokenizer.xml"
            ov.save_model(ov_tokenizer, str(tokenizer_path))
            print("✓ OpenVINO tokenizer created (openvino_tokenizer.xml/.bin)")
            print()
        except Exception as e:
            print(f"⚠️  Warning: Could not create OpenVINO tokenizer: {e}")
            print("   Model will still work with standard tokenizer")
            print()
        
    except Exception as e:
        print(f"❌ ERROR during conversion: {e}")
        return False
    
    # Success!
    print("=" * 60)
    print("✅ Embeddings Model Conversion Complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {out_dir}")
    print()
    
    return True


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    success = convert_embeddings()
    
    if not success:
        print()
        print("Conversion failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

