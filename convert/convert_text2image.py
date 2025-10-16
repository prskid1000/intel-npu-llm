#!/usr/bin/env python3
"""
OpenVINO Text2Image Model Conversion Script
Converts Stable Diffusion 1.5 for image generation using optimum-cli
Based on: https://docs.openvino.ai/2024/notebooks/text-to-image-genai-with-output.html
"""

import sys
import subprocess
from pathlib import Path
import warnings

def convert_text2image():
    """Convert Stable Diffusion 1.5 model to OpenVINO using optimum-cli"""
    
    print("=" * 60)
    print("Text2Image Model Conversion (Stable Diffusion 1.5)")
    print("=" * 60)
    print()
    
    model_id = "runwayml/stable-diffusion-v1-5"
    out_dir = Path("models/Text2Image/sd-1.5")
    
    print(f"Source Model: {model_id}")
    print(f"Output Directory: {out_dir}")
    print(f"Task: text-to-image")
    print(f"Method: optimum-cli export (OpenVINO GenAI compatible)")
    print()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already converted (look for model_index.json and tokenizers)
    if (out_dir / "model_index.json").exists():
        print("✓ Model already converted (found model_index.json)")
        print()
        
        # Check if both tokenizers exist (required for GenAI)
        tokenizer_xml = out_dir / "tokenizer" / "openvino_tokenizer.xml"
        clip_tokenizer_xml = out_dir / "openvino_tokenizer.xml"
        
        if tokenizer_xml.exists() and clip_tokenizer_xml.exists():
            print("✓ OpenVINO tokenizers found (Diffusers + CLIP)")
            print()
            return True
        else:
            print("⚠️  OpenVINO tokenizers missing, will regenerate...")
            if not tokenizer_xml.exists():
                print("   - Missing: tokenizer/openvino_tokenizer.xml")
            if not clip_tokenizer_xml.exists():
                print("   - Missing: openvino_tokenizer.xml (CLIP)")
            # Remove the model to force full reconversion with tokenizers
            import shutil
            print("   Removing old model files...")
            for subdir in ["text_encoder", "unet", "vae_decoder", "vae_encoder", "tokenizer", "clip_tokenizer"]:
                subdir_path = out_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
            for file in ["openvino_tokenizer.xml", "openvino_tokenizer.bin", "openvino_detokenizer.xml", "openvino_detokenizer.bin"]:
                file_path = out_dir / file
                if file_path.exists():
                    file_path.unlink()
            print("   Old model files removed")
            print()
    
    print("Converting Stable Diffusion pipeline using optimum-cli...")
    print("This includes: UNet, VAE, Text Encoder, Safety Checker, Tokenizer")
    print("This may take 5-10 minutes...")
    print()
    
    try:
        # Use optimum-cli export as recommended in OpenVINO docs
        # This properly generates openvino-tokenizers for GenAI pipeline
        import os
        optimum_cli = os.path.join(Path(sys.executable).parent, "optimum-cli.exe")
        
        cmd = [
            optimum_cli, "export", "openvino",
            "--model", model_id,
            "--task", "text-to-image",
            "--library", "diffusers",
            "--weight-format", "fp16",
            str(out_dir)
        ]
        
        print(f"Running optimum-cli export...")
        print()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print()
        print("✓ Stable Diffusion pipeline converted successfully")
        print()
        
        # Verify tokenizer was created
        tokenizer_xml = out_dir / "tokenizer" / "openvino_tokenizer.xml"
        tokenizer_bin = out_dir / "tokenizer" / "openvino_tokenizer.bin"
        detokenizer_xml = out_dir / "tokenizer" / "openvino_detokenizer.xml"
        
        if tokenizer_xml.exists() and tokenizer_bin.exists():
            print("✓ OpenVINO tokenizer generated successfully")
            print(f"   {tokenizer_xml}")
            print(f"   {tokenizer_bin}")
            if detokenizer_xml.exists():
                print(f"   {detokenizer_xml} (detokenizer)")
        else:
            print("⚠️  Warning: OpenVINO tokenizer files not found")
            print("   Model may not work with Text2ImagePipeline")
        
        print()
        
        # Convert CLIP tokenizer separately (required for GenAI Text2ImagePipeline)
        # Reference: https://medium.com/openvino-toolkit/stable-diffusion-controlnet-pipeline-with-openvino-in-c-c7bb306cc78e
        try:
            print("Converting CLIP tokenizer using convert_tokenizer...")
            import os
            convert_tokenizer_exe = os.path.join(Path(sys.executable).parent, "convert_tokenizer.exe")
            
            clip_tokenizer_dir = out_dir / "clip_tokenizer"
            
            cmd = [
                convert_tokenizer_exe,
                "openai/clip-vit-large-patch14",
                "--with-detokenizer",
                "-o", str(clip_tokenizer_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(result.stdout)
            
            # Copy CLIP tokenizer to root directory for Text2ImagePipeline
            import shutil
            if (clip_tokenizer_dir / "openvino_tokenizer.xml").exists():
                shutil.copy2(clip_tokenizer_dir / "openvino_tokenizer.xml", out_dir / "openvino_tokenizer.xml")
                shutil.copy2(clip_tokenizer_dir / "openvino_tokenizer.bin", out_dir / "openvino_tokenizer.bin")
                shutil.copy2(clip_tokenizer_dir / "openvino_detokenizer.xml", out_dir / "openvino_detokenizer.xml")
                shutil.copy2(clip_tokenizer_dir / "openvino_detokenizer.bin", out_dir / "openvino_detokenizer.bin")
                print("✓ CLIP tokenizer copied to root directory")
                print(f"   {out_dir / 'openvino_tokenizer.xml'}")
                print(f"   {out_dir / 'openvino_tokenizer.bin'}")
            else:
                print("⚠️  Warning: CLIP tokenizer conversion failed")
            
            print()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not convert CLIP tokenizer: {e}")
            print("   Model may not work properly with Text2ImagePipeline")
            print()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR during conversion: {e}")
        print()
        print("Make sure you have installed:")
        print("  pip install optimum-intel[openvino,diffusers]")
        print("  pip install openvino openvino-tokenizers")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    # Success!
    print("=" * 60)
    print("✅ Text2Image Model Conversion Complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {out_dir}")
    print()
    print("The model is now compatible with ov_genai.Text2ImagePipeline")
    print("Note: Works on both CPU and GPU")
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
