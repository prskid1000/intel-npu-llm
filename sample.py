def main():
    import openvino_genai as ov_genai

    import openvino as ov
    core = ov.Core()
    devices = core.available_devices()
    print("Available devices:", devices)
    
    model_path = "Qwen/Qwen2.5-3B"
    pipe = ov_genai.LLMPipeline(model_path, "NPU")
    print(pipe.generate("The Sun is yellow because", max_new_tokens=100))


if '__main__' == __name__:
    main()