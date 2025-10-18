#!/usr/bin/env python3
"""
OpenVINO DeepSeek-VL Model Conversion Script
Based on: https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat

This script converts DeepSeek-VL-1.3b-chat to OpenVINO format with INT4 quantization
for optimal NPU performance.
"""

import sys
from pathlib import Path
import warnings

def convert_deepseek_vl():
    """Convert DeepSeek-VL-1.3b-chat model to OpenVINO INT4 format"""
    
    print("=" * 60)
    print("DeepSeek-VL-1.3b-chat OpenVINO Conversion")
    print("=" * 60)
    print()
    
    model_id = "deepseek-ai/deepseek-vl-1.3b-chat"
    out_dir = Path("models/DeepSeek/deepseek-vl-1.3b-chat")
    
    print(f"Source Model: {model_id}")
    print(f"Output Directory: {out_dir}")
    print(f"Quantization: INT4 symmetric (group-size 128)")
    print()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.intel.openvino import OVModelForVisualCausalLM
        import nncf
        import torch
        import gc
        
        print("✓ Required packages loaded")
        print()
        
    except ImportError as e:
        print(f"❌ ERROR: Missing required package: {e}")
        print()
        print("Please ensure you have installed:")
        print("  pip install optimum-intel[openvino] nncf")
        return False
    
    # Step 1: Load and convert base model
    print("[1/3] Converting base model to OpenVINO FP16...")
    print("      This may take several minutes...")
    print()
    
    try:
        # Check if already converted
        if (out_dir / "openvino_language_model.xml").exists():
            print("✓ Model already converted (found in output directory)")
            print()
        else:
            # Convert with FP16 first
            model = OVModelForVisualCausalLM.from_pretrained(
                model_id,
                export=True,
                trust_remote_code=True,
                compile=False,
                load_in_8bit=False,
            )
            
            # Save FP16 model
            model.save_pretrained(out_dir)
            
            # Clear memory
            del model
            gc.collect()
            
            print("✓ Base model converted to OpenVINO FP16")
            print()
        
    except Exception as e:
        print(f"❌ ERROR during base conversion: {e}")
        print()
        print("Note: DeepSeek-VL uses custom architecture with trust_remote_code=True")
        print("      Make sure you have the latest optimum-intel version")
        return False
    
    # Step 1.5: Download tokenizer and processor files + convert to OpenVINO
    print("[1.5/3] Setting up tokenizer and processor...")
    print()
    
    try:
        from transformers import AutoTokenizer, AutoProcessor
        from openvino_tokenizers import convert_tokenizer
        import json
        
        # Download and save tokenizer
        print("   Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(out_dir)
        
        # Download and save processor (includes VLChatProcessor)
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            processor.save_pretrained(out_dir)
            print("   ✓ Processor files saved (includes vision processor)")
        except Exception as e:
            print(f"   Note: Could not save processor: {e}")
            print("   Attempting to save image processor separately...")
            try:
                from transformers import AutoImageProcessor
                image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
                image_processor.save_pretrained(out_dir)
                print("   ✓ Image processor saved")
            except Exception as e2:
                print(f"   Warning: Could not save image processor: {e2}")
        
        print("   ✓ HuggingFace tokenizer files saved")
        
        # Add chat template if missing
        tokenizer_config_path = out_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)
            
            if 'chat_template' not in tokenizer_config:
                print("   Adding chat template...")
                # DeepSeek-VL uses User/Assistant format
                tokenizer_config['chat_template'] = "{% for message in messages %}{% if message['role'] == 'User' %}User: {{ message['content'] }}\n\n{% elif message['role'] == 'Assistant' %}Assistant: {{ message['content'] }}\n\n{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"
                
                with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                    json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
                print("   ✓ Chat template added")
        
        # Convert tokenizer and detokenizer to OpenVINO format
        print("   Converting tokenizer to OpenVINO format...")
        import openvino as ov
        
        # Create tokenizer
        ov_tokenizer = convert_tokenizer(tokenizer)
        tokenizer_path = out_dir / "openvino_tokenizer.xml"
        ov.save_model(ov_tokenizer, str(tokenizer_path))
        print("   ✓ OpenVINO tokenizer created (openvino_tokenizer.xml/.bin)")
        
        # Create detokenizer
        print("   Converting detokenizer to OpenVINO format...")
        # convert_tokenizer with with_detokenizer=True returns a tuple (tokenizer, detokenizer)
        ov_tokenizer_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True, skip_special_tokens=True)
        detokenizer_path = out_dir / "openvino_detokenizer.xml"
        # Extract the detokenizer (second element of the tuple)
        ov.save_model(ov_tokenizer_detokenizer[1], str(detokenizer_path))
        print("   ✓ OpenVINO detokenizer created (openvino_detokenizer.xml/.bin)")
        print()
        
    except Exception as e:
        print(f"⚠️  Warning: Could not setup tokenizer: {e}")
        print("   The model may not work properly without tokenizer files")
        print()
    
    # Step 2: Apply INT4 quantization
    print("[2/3] Applying INT4 quantization...")
    print("      This optimizes the model for NPU inference")
    print()
    
    try:
        # Re-export with quantization directly
        print("   Quantization settings:")
        print("   - Mode: INT4 symmetric")
        print("   - Group size: 128")
        print("   - Ratio: 1.0 (all weights quantized)")
        print()
        
        # Export with quantization in one step
        model = OVModelForVisualCausalLM.from_pretrained(
            model_id,
            export=True,
            trust_remote_code=True,
            compile=False,
            load_in_8bit=False,
            quantization_config={
                "bits": 4,
                "sym": True,
                "group_size": 128,
                "ratio": 1.0,
            }
        )
        
        # Save quantized model
        model.save_pretrained(out_dir)
        
        print("✓ INT4 quantization applied and saved")
        print()
        
    except Exception as e:
        print(f"❌ ERROR during quantization: {e}")
        print()
        print("Attempting alternative quantization method...")
        
        try:
            # Alternative: use compress_weights on the existing model
            from optimum.intel import OVQuantizer
            
            quantizer = OVQuantizer.from_pretrained(out_dir, trust_remote_code=True)
            
            quantization_config = {
                "algorithm": "weight_compression",
                "mode": "INT4_SYM",
                "group_size": 128,
                "ratio": 1.0,
            }
            
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=out_dir
            )
            
            print("✓ INT4 quantization applied using alternative method")
            print()
            
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            print()
            print("⚠️  Keeping FP16 model (no quantization)")
            print("    This will work but use more memory than INT4")
            print()
    
    # Step 3: Save model info
    print("[3/3] Saving model information...")
    print()
    
    try:
        import json
        
        model_info = {
            "model_id": model_id,
            "model_type": "vision-language",
            "architecture": "DeepSeek-VL",
            "vision_encoder": "SigLIP-L",
            "image_size": "384x384",
            "base_model": "DeepSeek-LLM-1.3b-base",
            "quantization": "INT4 symmetric",
            "framework": "OpenVINO",
            "usage": "Real-world vision and language understanding",
            "capabilities": [
                "Logical diagrams",
                "Web pages",
                "Formula recognition",
                "Scientific literature",
                "Natural images",
                "Embodied intelligence"
            ]
        }
        
        info_path = out_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print("✓ Model information saved")
        print()
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save model info: {e}")
        print()
    
    # Success!
    print("=" * 60)
    print("✅ Conversion Complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {out_dir}")
    print()
    print("Model capabilities:")
    print("  • Image size: 384x384 pixels")
    print("  • Vision encoder: SigLIP-L")
    print("  • Language model: DeepSeek-LLM-1.3b")
    print("  • Training: ~400B vision-language tokens")
    print()
    print("Next steps:")
    print("  1. Update your config.json to use this model")
    print("  2. Run: python npu.py")
    print()
    print("Example config entry:")
    print("  {")
    print('    "id": "deepseek-vl-1.3b-chat",')
    print('    "name": "DeepSeek-VL 1.3B Chat",')
    print('    "path": "models/DeepSeek/deepseek-vl-1.3b-chat",')
    print('    "type": "vision-language"')
    print("  }")
    print()
    
    return True


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    success = convert_deepseek_vl()
    
    if not success:
        print()
        print("Conversion failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

