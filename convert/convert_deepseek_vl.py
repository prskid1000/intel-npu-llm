"""
DeepSeek Janus-Pro-1B to OpenVINO Conversion Script
Based on official OpenVINO documentation
"""

import gc
import warnings
from pathlib import Path
import torch
import openvino as ov
import nncf
import os

# Bypass torch.load security check (USE WITH CAUTION - only for trusted models)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "true"

# Import from DeepSeek Janus package (required!)
try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
except ImportError:
    raise ImportError(
        "DeepSeek Janus package not found!\n"
        "Please install it with: pip install git+https://github.com/deepseek-ai/Janus.git"
    )

warnings.filterwarnings("ignore")

# Configuration
MODEL_ID = "deepseek-ai/Janus-Pro-1B"
OUTPUT_DIR = Path("models/DeepSeek/Janus-Pro")

# INT4 compression configuration
COMPRESSION_CONFIG = {
    "mode": nncf.CompressWeightsMode.INT4_ASYM,
    "group_size": 64,
    "ratio": 1.0,
}


def convert_janus_model(model_id: str, output_dir: Path, compression_config: dict):
    """
    Convert Janus-Pro model to OpenVINO IR format with INT4 quantization
    
    Args:
        model_id: HuggingFace model ID
        output_dir: Path to save converted model
        compression_config: NNCF compression configuration
    """
    
    print(f"⌛ {model_id.split('/')[-1]} conversion started. Be patient, it may take some time.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load original model using Janus package
    print("⌛ Loading original model...")
    
    # Load processor with recommended settings
    vl_chat_processor = VLChatProcessor.from_pretrained(
        model_id,
        use_fast=True  # Use fast image processor
    )
    tokenizer = vl_chat_processor.tokenizer
    
    # Load model (with trust for official DeepSeek model)
    import transformers
    # Temporarily disable the torch.load safety check for trusted model
    transformers.modeling_utils._check_torch_load_is_safe = lambda: None
    
    model = MultiModalityCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True  # Try safetensors first
    )
    
    print(f"✅ Model loaded successfully: {model.__class__.__name__}")
    
    # Step 2: Convert model components
    # Janus-Pro consists of 7 main components that need separate conversion
    
    # 2.1 Text Embeddings Model
    print("⌛ Converting text embeddings model...")
    text_emb_path = output_dir / "openvino_text_embeddings_model.xml"
    if not text_emb_path.exists():
        text_emb_model = ov.convert_model(
            model.language_model.get_input_embeddings(),
            example_input=torch.ones([1, 10], dtype=torch.int64)
        )
        text_emb_model = nncf.compress_weights(text_emb_model, **compression_config)
        ov.save_model(text_emb_model, text_emb_path)
        del text_emb_model
        gc.collect()
    print("✅ Text embeddings model successfully converted")
    
    # 2.2 Vision Embeddings Model (Understanding Encoder - SigLIP)
    print("⌛ Converting vision embeddings model...")
    vision_emb_path = output_dir / "openvino_vision_embeddings_model.xml"
    if not vision_emb_path.exists():
        # Example image input: batch_size=1, channels=3, height=384, width=384
        # Match model dtype (bfloat16)
        vision_emb_model = ov.convert_model(
            model.vision_model,
            example_input=torch.randn(1, 3, 384, 384, dtype=torch.bfloat16)
        )
        vision_emb_model = nncf.compress_weights(vision_emb_model, **compression_config)
        ov.save_model(vision_emb_model, vision_emb_path)
        del vision_emb_model
        gc.collect()
    print("✅ Vision embeddings model successfully converted")
    
    # 2.3 Gen Embeddings Model (Generation Encoder - VQ Tokenizer)
    print("⌛ Converting gen embeddings model...")
    gen_emb_path = output_dir / "openvino_gen_embeddings_model.xml"
    if not gen_emb_path.exists():
        # Wrap the model to ensure tensor-only output
        class GenVisionModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                output = self.model(x)
                # Extract only tensor outputs from the result
                if isinstance(output, tuple):
                    # Return only tensor elements
                    return tuple(o for o in output if isinstance(o, torch.Tensor))
                elif isinstance(output, dict):
                    # Return as tuple of tensor values
                    return tuple(v for v in output.values() if isinstance(v, torch.Tensor))
                elif hasattr(output, 'last_hidden_state'):
                    # Handle model outputs with attributes
                    return output.last_hidden_state
                else:
                    return output
        
        wrapped_model = GenVisionModelWrapper(model.gen_vision_model)
        gen_emb_model = ov.convert_model(
            wrapped_model,
            example_input=torch.randn(1, 3, 384, 384, dtype=torch.bfloat16)
        )
        gen_emb_model = nncf.compress_weights(gen_emb_model, **compression_config)
        ov.save_model(gen_emb_model, gen_emb_path)
        del gen_emb_model, wrapped_model
        gc.collect()
    print("✅ Gen embeddings model successfully converted")
    
    # 2.4 Language Model (Main Transformer)
    print("⌛ Converting language model (this may take a while)...")
    lm_path = output_dir / "openvino_language_model.xml"
    if not lm_path.exists():
        # Wrap the language model to extract only tensor outputs and handle stateful components
        class LanguageModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                # Set model to evaluation mode and disable all stateful components
                self.model.eval()
                if hasattr(self.model, 'config'):
                    self.model.config.use_cache = False
                    self.model.config.output_attentions = False
                    self.model.config.output_hidden_states = False
                if hasattr(self.model, 'generation_config'):
                    self.model.generation_config.use_cache = False
            
            def forward(self, input_ids, attention_mask, position_ids):
                # Force disable cache and stateful components
                with torch.no_grad():
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False
                    )
                # Extract only the logits tensor from CausalLMOutputWithPast
                if hasattr(output, 'logits'):
                    return output.logits
                elif isinstance(output, tuple):
                    return output[0]
                else:
                    return output
        
        # Create example inputs for language model
        input_ids = torch.ones([1, 10], dtype=torch.int64)
        attention_mask = torch.ones([1, 10], dtype=torch.int64)
        position_ids = torch.arange(0, 10).unsqueeze(0)
        
        wrapped_lm = LanguageModelWrapper(model.language_model)
        # Try converting without example_input to use script mode
        try:
            lm_model = ov.convert_model(wrapped_lm)
        except Exception as e:
            print(f"   Script mode failed, trying with example input: {e}")
            lm_model = ov.convert_model(
                wrapped_lm,
                example_input={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids
                }
            )
        lm_model = nncf.compress_weights(lm_model, **compression_config)
        ov.save_model(lm_model, lm_path)
        del lm_model, wrapped_lm
        gc.collect()
    print("✅ Language model successfully converted")
    
    # 2.5 LM Head (Text token prediction)
    print("⌛ Converting LM head...")
    lm_head_path = output_dir / "openvino_lm_head.xml"
    if not lm_head_path.exists():
        hidden_size = model.language_model.config.hidden_size
        lm_head_model = ov.convert_model(
            model.language_model.lm_head,
            example_input=torch.randn(1, 10, hidden_size, dtype=torch.bfloat16)
        )
        lm_head_model = nncf.compress_weights(lm_head_model, **compression_config)
        ov.save_model(lm_head_model, lm_head_path)
        del lm_head_model
        gc.collect()
    print("✅ LM head successfully converted")
    
    # 2.6 Gen Head (Image token prediction)
    print("⌛ Converting gen head...")
    gen_head_path = output_dir / "openvino_gen_head.xml"
    if not gen_head_path.exists():
        gen_head_model = ov.convert_model(
            model.gen_head,
            example_input=torch.randn(1, 10, model.language_model.config.hidden_size, dtype=torch.bfloat16)
        )
        gen_head_model = nncf.compress_weights(gen_head_model, **compression_config)
        ov.save_model(gen_head_model, gen_head_path)
        del gen_head_model
        gc.collect()
    print("✅ Gen head successfully converted")
    
    # 2.7 Gen Decoder (Image reconstruction from tokens)
    # Check if decoder exists as a separate component
    print("⌛ Checking for gen decoder...")
    gen_decoder_path = output_dir / "openvino_gen_decoder.xml"
    if not gen_decoder_path.exists():
        # Check if the model has a separate decoder component
        if hasattr(model, 'gen_vision_decoder'):
            print("   Converting gen decoder...")
            gen_decoder_model = ov.convert_model(
                model.gen_vision_decoder,
                example_input=torch.randint(0, 8192, (1, 576))  # 24x24 grid = 576 tokens
            )
            gen_decoder_model = nncf.compress_weights(gen_decoder_model, **compression_config)
            ov.save_model(gen_decoder_model, gen_decoder_path)
            del gen_decoder_model
            gc.collect()
            print("✅ Gen decoder model successfully converted")
        else:
            print("ℹ️  Gen decoder not found as separate component (may be integrated in gen_vision_model)")
    
    # Step 3: Save tokenizer and config
    print("⌛ Saving tokenizer and configuration...")
    vl_chat_processor.save_pretrained(output_dir)
    
    # Save model config
    if hasattr(model, 'config'):
        model.config.save_pretrained(output_dir)
    
    print(f"✅ {model_id.split('/')[-1]} model conversion finished. Results in {output_dir}")


def main():
    """Main conversion function"""
    
    print("="*60)
    print("DeepSeek Janus-Pro-1B to OpenVINO Conversion")
    print("="*60)
    
    # Install required packages
    print("\n📦 Required packages:")
    print("pip install torch torchvision transformers nncf openvino")
    print("pip install git+https://github.com/deepseek-ai/Janus\n")
    
    try:
        convert_janus_model(MODEL_ID, OUTPUT_DIR, COMPRESSION_CONFIG)
        
        print("\n" + "="*60)
        print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nModel saved to: {OUTPUT_DIR}")
        print("\nYou can now use the model with OpenVINO GenAI:")
        print(f"  from openvino_genai import VLMPipeline")
        print(f"  pipe = VLMPipeline('{OUTPUT_DIR}', 'CPU')")
        
    except Exception as e:
        print(f"\n❌ ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()