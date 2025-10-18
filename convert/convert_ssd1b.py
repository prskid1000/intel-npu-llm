#!/usr/bin/env python3
"""
OpenVINO SSD-1B Model Conversion Script
Converts Segmind Stable Diffusion 1B for image generation using optimum-cli
Model: https://huggingface.co/segmind/SSD-1B
Based on: Distilled SDXL (50% smaller, 60% faster)
"""

import sys
import subprocess
from pathlib import Path
import warnings

def convert_ssd1b():
    """Convert SSD-1B model to OpenVINO using optimum-cli"""
    
    print("=" * 60)
    print("SSD-1B Model Conversion (Segmind Stable Diffusion 1B)")
    print("=" * 60)
    print()
    
    model_id = "segmind/SSD-1B"
    out_dir = Path("models/Text2Image/SSD-1B")
    
    print(f"Source Model: {model_id}")
    print(f"Output Directory: {out_dir}")
    print(f"Task: text-to-image")
    print(f"Method: optimum-cli export (OpenVINO GenAI compatible)")
    print(f"Architecture: Distilled SDXL (1.3B parameters)")
    print()
    print("Note: SSD-1B is 50% smaller and 60% faster than SDXL")
    print("      Best results with CFG ~9.0 and negative prompting")
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
            for subdir in ["text_encoder", "text_encoder_2", "unet", "vae_decoder", "vae_encoder", "tokenizer", "tokenizer_2", "clip_tokenizer"]:
                subdir_path = out_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
            for file in ["openvino_tokenizer.xml", "openvino_tokenizer.bin", "openvino_detokenizer.xml", "openvino_detokenizer.bin"]:
                file_path = out_dir / file
                if file_path.exists():
                    file_path.unlink()
            print("   Old model files removed")
            print()
    
    print("Converting SSD-1B pipeline using optimum-cli...")
    print("This includes: UNet, VAE, Text Encoders, Tokenizers")
    print("⚠️  This may take 10-20 minutes and requires ~8GB disk space...")
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
        print("✓ SSD-1B pipeline converted successfully")
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
        # SSD-1B uses SDXL's dual text encoder architecture
        try:
            print("Converting CLIP tokenizer using convert_tokenizer...")
            import os
            convert_tokenizer_exe = os.path.join(Path(sys.executable).parent, "convert_tokenizer.exe")
            
            clip_tokenizer_dir = out_dir / "clip_tokenizer"
            
            # SDXL uses the same CLIP tokenizer as SD 1.5
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
        print()
        print("Note: SSD-1B requires sufficient memory for download and conversion")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    # Success!
    print("=" * 60)
    print("✅ SSD-1B Model Conversion Complete!")
    print("=" * 60)
    print()
    print(f"Model saved to: {out_dir}")
    print()
    print("The model is now compatible with ov_genai.Text2ImagePipeline")
    print("Note: Works on both CPU and NPU (Neural Processing Unit)")
    print()
    print("Usage Tips:")
    print("  - Use negative prompts for best quality")
    print("  - Recommended CFG scale: ~9.0")
    print("  - Supports multiple resolutions (1024x1024, 1152x896, etc.)")
    print("  - 60% faster inference than SDXL")
    print()
    
    return True


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    success = convert_ssd1b()
    
    if not success:
        print()
        print("Conversion failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

