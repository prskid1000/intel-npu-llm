"""
Multimodal RAG Example - OpenVINO GenAI API
Demonstrates file upload, image processing, and RAG capabilities
"""

import base64
from openai import OpenAI
from pathlib import Path

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 data URI"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Detect image format
    ext = Path(image_path).suffix.lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }.get(ext, 'image/jpeg')
    
    return f"data:{mime_type};base64,{encoded}"


def example_file_upload():
    """Example: Upload a file"""
    print("=" * 60)
    print("Example 1: File Upload")
    print("=" * 60)
    
    # Create a sample text file
    with open("sample_document.txt", "w") as f:
        f.write("""
Title: OpenVINO Overview

OpenVINO (Open Visual Inference and Neural Network Optimization) is an open-source toolkit 
for optimizing and deploying AI inference. It enables deep learning models to run faster 
on various hardware platforms including CPUs, GPUs, and NPUs.

Key Features:
- Model optimization and quantization
- Multi-platform support
- High performance inference
- Support for various AI frameworks
        """)
    
    # Upload the file
    with open("sample_document.txt", "rb") as f:
        file_obj = client.files.create(
            file=f,
            purpose="assistants"
        )
    
    print(f"✅ File uploaded successfully!")
    print(f"   File ID: {file_obj.id}")
    print(f"   Filename: {file_obj.filename}")
    print(f"   Size: {file_obj.bytes} bytes")
    print()
    
    return file_obj.id


def example_list_files():
    """Example: List uploaded files"""
    print("=" * 60)
    print("Example 2: List Uploaded Files")
    print("=" * 60)
    
    files = client.files.list()
    print(f"Total files: {len(files.data)}")
    for file in files.data:
        print(f"  - {file.filename} (ID: {file.id}, {file.bytes} bytes)")
    print()


def example_rag_with_file(file_id: str):
    """Example: RAG - Ask questions about uploaded document"""
    print("=" * 60)
    print("Example 3: RAG - Question Answering from Document")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided documents."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is OpenVINO and what are its key features?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": file_id  # Reference uploaded file
                        }
                    }
                ]
            }
        ],
        max_tokens=200
    )
    
    print(f"Question: What is OpenVINO and what are its key features?")
    print(f"Answer: {response.choices[0].message.content}")
    print()


def example_image_base64():
    """Example: Vision model with base64 image"""
    print("=" * 60)
    print("Example 4: Vision Model with Base64 Image")
    print("=" * 60)
    print("⚠️  Requires a VLM model configured in config.json")
    print()
    
    # Example with a hypothetical image
    # In real usage, you would:
    # 1. Have an actual image file
    # 2. Have a VLM model configured
    
    # image_base64 = encode_image_to_base64("path/to/your/image.jpg")
    # 
    # response = client.chat.completions.create(
    #     model="your-vlm-model",  # e.g., "minicpm-v"
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": "What do you see in this image?"
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": image_base64
    #                     }
    #                 }
    #             ]
    #         }
    #     ],
    #     max_tokens=300
    # )
    # 
    # print(f"Response: {response.choices[0].message.content}")
    print("See code for usage example")
    print()


def example_image_url():
    """Example: Vision model with image URL"""
    print("=" * 60)
    print("Example 5: Vision Model with Image URL")
    print("=" * 60)
    print("⚠️  Requires a VLM model configured in config.json")
    print()
    
    # Example with a remote image URL
    # response = client.chat.completions.create(
    #     model="your-vlm-model",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": "Describe this image"
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": "https://example.com/image.jpg"
    #                     }
    #                 }
    #             ]
    #         }
    #     ],
    #     max_tokens=300
    # )
    print("See code for usage example")
    print()


def example_multimodal_conversation():
    """Example: Multi-turn conversation with files and images"""
    print("=" * 60)
    print("Example 6: Multi-turn Multimodal Conversation")
    print("=" * 60)
    
    # Create a technical document
    with open("technical_doc.txt", "w") as f:
        f.write("""
Technical Specification: Neural Processing Unit (NPU)

The NPU is a specialized processor designed for AI acceleration.

Performance Metrics:
- Peak Performance: 40 TOPS
- Power Efficiency: 15 TOPS/W
- Memory Bandwidth: 136 GB/s

Supported Operations:
- Matrix multiplication
- Convolution
- Activation functions
- Quantization
        """)
    
    # Upload the document
    with open("technical_doc.txt", "rb") as f:
        doc_file = client.files.create(file=f, purpose="assistants")
    
    print(f"Uploaded document: {doc_file.filename}")
    
    # Multi-turn conversation
    messages = [
        {
            "role": "system",
            "content": "You are a technical AI assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please read this technical document."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": doc_file.id}
                }
            ]
        }
    ]
    
    # First query
    response1 = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=messages,
        max_tokens=100
    )
    
    print(f"\n👤 User: Please read this technical document.")
    print(f"🤖 Assistant: {response1.choices[0].message.content}")
    
    # Continue conversation
    messages.append({
        "role": "assistant",
        "content": response1.choices[0].message.content
    })
    messages.append({
        "role": "user",
        "content": "What is the power efficiency of the NPU?"
    })
    
    response2 = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=messages,
        max_tokens=100
    )
    
    print(f"\n👤 User: What is the power efficiency of the NPU?")
    print(f"🤖 Assistant: {response2.choices[0].message.content}")
    print()


def example_delete_file(file_id: str):
    """Example: Delete uploaded file"""
    print("=" * 60)
    print("Example 7: Delete File")
    print("=" * 60)
    
    result = client.files.delete(file_id)
    print(f"✅ File deleted: {file_id}")
    print()


def main():
    """Run all examples"""
    print("=" * 60)
    print("Multimodal RAG Examples - OpenVINO GenAI API")
    print("=" * 60)
    print()
    
    try:
        # File upload examples
        file_id = example_file_upload()
        example_list_files()
        example_rag_with_file(file_id)
        
        # Vision examples (commented out as they require VLM models)
        example_image_base64()
        example_image_url()
        
        # Advanced examples
        example_multimodal_conversation()
        
        # Cleanup
        example_delete_file(file_id)
        
        print("=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  1. The server is running: python npu.py")
        print("  2. Models are properly configured in config.json")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

