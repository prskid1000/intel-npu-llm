#!/usr/bin/env python3
"""
OpenVINO Moderation Model Conversion Script
Converts Toxic-BERT for content safety/moderation
"""

import sys
from pathlib import Path
import warnings

def convert_moderation():
    """Convert Toxic-BERT moderation model to OpenVINO"""
    
    print("=" * 60)
    print("Moderation Model Conversion (Toxic-BERT)")
    print("=" * 60)
    print()
    
    model_id = "unitary/toxic-bert"
    out_dir = Path("models/Moderation/toxic-bert")
    
    print(f"Source Model: {model_id}")
    print(f"Output Directory: {out_dir}")
    print(f"Task: Text classification (toxicity detection)")
    print()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.intel import OVModelForSequenceClassification
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
        # Convert model
        model = OVModelForSequenceClassification.from_pretrained(
            model_id,
            export=True,
            compile=False,
        )
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Save both
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        
        print("✓ Model converted successfully")
        print()
        
    except Exception as e:
        print(f"❌ ERROR during conversion: {e}")
        return False
    
    # Success!
    print("=" * 60)
    print("✅ Moderation Model Conversion Complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {out_dir}")
    print()
    
    return True


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    success = convert_moderation()
    
    if not success:
        print()
        print("Conversion failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

