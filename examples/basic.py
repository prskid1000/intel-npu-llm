def main():
    import openvino_genai as ov_genai
    import openvino as ov
    import numpy as np
    
    core = ov.Core()
    print("Available devices:", core.available_devices)
    
    # Using Phi-3.5-Vision (VLM) for text generation
    model_path = "./models/Phi/Phi-3.5-vision-instruct"
    pipe = ov_genai.VLMPipeline(model_path, "NPU")
    
    # VLM on NPU requires an image input, so create a dummy image for text-only mode
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_tensor = ov.Tensor(dummy_image)
    
    # Generate with dummy image
    result = pipe.generate("The Sun is yellow because", image=dummy_tensor, max_new_tokens=100)
    response_text = result if isinstance(result, str) else result.texts[0]
    
    print(response_text)


if __name__ == '__main__':
    main()