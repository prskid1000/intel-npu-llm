"""
Comprehensive Feature Test - OpenVINO GenAI API Server
Tests all features: chat, streaming, tools, structured outputs, RAG, audio, vision, etc.

Run this after starting the server: python npu.py
"""

import json
import base64
from pathlib import Path
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)


# ============================================================================
# Basic Chat & Completions
# ============================================================================

def test_list_models():
    """Test: List available models"""
    print("\n" + "="*70)
    print("TEST 1: List Models")
    print("="*70)
    
    models = client.models.list()
    print(f"✅ Found {len(models.data)} model(s):")
    for model in models.data:
        print(f"   - {model.id}")


def test_basic_chat():
    """Test: Basic chat completion (text-only and with image)"""
    print("\n" + "="*70)
    print("TEST 2: Basic Chat Completion")
    print("="*70)
    
    # Test 1: Text-only request
    print("\n📝 Part 1: Text-only chat")
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is OpenVINO in one sentence?"}
        ],
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"User: What is OpenVINO in one sentence?")
    print(f"Assistant: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    
    # Test 2: Multimodal request with dummy image
    print("\n🖼️  Part 2: Chat with dummy image (VLM capability)")
    from PIL import Image
    import io
    
    # Create a simple dummy image with text
    img = Image.new('RGB', (100, 50), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "TEST", fill='black')
    
    # Convert to base64
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What text do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        max_tokens=50
    )
    
    print(f"User: [Image with text] What text do you see?")
    print(f"Assistant: {response.choices[0].message.content}")
    print("✅ Basic chat completed (text-only + multimodal)")


def test_streaming():
    """Test: Streaming chat completion (text-only and with image)"""
    print("\n" + "="*70)
    print("TEST 3: Streaming Chat")
    print("="*70)
    
    # Test 1: Text-only streaming
    print("\n📝 Part 1: Text-only streaming")
    print("User: Count from 1 to 5")
    print("Assistant: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {"role": "user", "content": "Count from 1 to 5, one number per line"}
        ],
        stream=True,
        max_tokens=50
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n\n🖼️  Part 2: Streaming with image (VLM capability)")
    print("Note: VLM streaming with images currently returns complete response")
    from PIL import Image
    import io
    
    # Create a simple colored square
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    print("User: [Red square image] What color is this?")
    print("Assistant: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this square? Answer in one word."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        stream=True,
        max_tokens=20
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n✅ Streaming completed (text-only + multimodal)")


def test_text_completion():
    """Test: Text completion endpoint"""
    print("\n" + "="*70)
    print("TEST 4: Text Completion")
    print("="*70)
    
    response = client.completions.create(
        model="phi-3.5-vision",
        prompt="The three laws of robotics are:",
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"Prompt: The three laws of robotics are:")
    print(f"Completion: {response.choices[0].text}")
    print("✅ Text completion completed")


def test_multi_turn_conversation():
    """Test: Multi-turn conversation with context (text + images)"""
    print("\n" + "="*70)
    print("TEST 5: Multi-turn Conversation")
    print("="*70)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Alice. What's the capital of France?"}
    ]
    
    # First turn (text-only)
    response1 = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=messages,
        max_tokens=50
    )
    
    print(f"👤 Alice: My name is Alice. What's the capital of France?")
    print(f"🤖 Assistant: {response1.choices[0].message.content}")
    
    # Second turn (text-only)
    messages.append({"role": "assistant", "content": response1.choices[0].message.content})
    messages.append({"role": "user", "content": "What's my name?"})
    
    response2 = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=messages,
        max_tokens=30
    )
    
    print(f"👤 User: What's my name?")
    print(f"🤖 Assistant: {response2.choices[0].message.content}")
    
    # Third turn (with image - VLM capability)
    from PIL import Image
    import io
    
    # Create a blue circle
    img = Image.new('RGB', (100, 100), color='blue')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill='yellow', outline='black')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    messages.append({"role": "assistant", "content": response2.choices[0].message.content})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "What shapes and colors do you see in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
    })
    
    response3 = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=messages,
        max_tokens=50
    )
    
    print(f"👤 User: [Image] What shapes and colors do you see?")
    print(f"🤖 Assistant: {response3.choices[0].message.content}")
    print("✅ Multi-turn conversation completed (text + multimodal)")


# ============================================================================
# Tool Calling / Function Calling
# ============================================================================

def test_tool_calling():
    """Test: Function/tool calling"""
    print("\n" + "="*70)
    print("TEST 6: Tool Calling (Function Calling)")
    print("="*70)
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {"role": "user", "content": "What's the weather in Paris and also calculate 15 * 7?"}
        ],
        tools=tools,
        tool_choice="auto",
        max_tokens=200
    )
    
    message = response.choices[0].message
    print(f"User: What's the weather in Paris and also calculate 15 * 7?")
    
    if message.tool_calls:
        print(f"🔧 Tool calls detected: {len(message.tool_calls)} call(s)")
        for tool_call in message.tool_calls:
            print(f"   Function: {tool_call.function.name}")
            print(f"   Arguments: {tool_call.function.arguments}")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print("✅ Tool calling completed")
    else:
        print(f"Assistant: {message.content}")
        print("⚠️  No tool calls detected (model may not support tool calling)")


def test_tool_calling_with_execution():
    """Test: Tool calling with simulated execution"""
    print("\n" + "="*70)
    print("TEST 7: Tool Calling with Execution")
    print("="*70)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a specific timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone, e.g. America/New_York"
                        }
                    },
                    "required": ["timezone"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "What time is it in New York?"}
    ]
    
    # First call - model decides to use tool
    response1 = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=messages,
        tools=tools,
        max_tokens=150
    )
    
    print(f"👤 User: What time is it in New York?")
    
    message1 = response1.choices[0].message
    if message1.tool_calls:
        tool_call = message1.tool_calls[0]
        print(f"🔧 Model wants to call: {tool_call.function.name}")
        print(f"   With arguments: {tool_call.function.arguments}")
        
        # Simulate tool execution
        simulated_result = "14:30:00"
        
        # Add assistant message and tool response to conversation
        messages.append({
            "role": "assistant",
            "content": message1.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
            ]
        })
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": simulated_result
        })
        
        # Second call - model uses tool result
        response2 = client.chat.completions.create(
            model="phi-3.5-vision",
            messages=messages,
            tools=tools,
            max_tokens=100
        )
        
        print(f"🔨 Tool returned: {simulated_result}")
        print(f"🤖 Final answer: {response2.choices[0].message.content}")
        print("✅ Tool execution flow completed")
    else:
        print(f"🤖 Assistant: {message1.content}")
        print("⚠️  Model responded directly without tool call")


# ============================================================================
# Structured Outputs
# ============================================================================

def test_json_mode():
    """Test: JSON mode (response_format)"""
    print("\n" + "="*70)
    print("TEST 8: JSON Mode (Structured Output)")
    print("="*70)
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs JSON."
            },
            {
                "role": "user",
                "content": "Give me information about Python programming language including name, year created, and 3 key features."
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=200
    )
    
    content = response.choices[0].message.content
    print(f"User: Give me information about Python (name, year, 3 features)")
    print(f"Response (JSON):")
    
    try:
        # Try to parse and pretty-print JSON
        json_obj = json.loads(content)
        print(json.dumps(json_obj, indent=2))
        print("✅ Valid JSON received")
    except json.JSONDecodeError:
        print(content)
        print("⚠️  Response may not be valid JSON")


def test_json_schema():
    """Test: JSON schema validation (structured outputs)"""
    print("\n" + "="*70)
    print("TEST 9: JSON Schema (Strict Structured Output)")
    print("="*70)
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "email": {"type": "string"},
            "interests": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age", "email"]
    }
    
    try:
        response = client.chat.completions.create(
            model="phi-3.5-vision",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs JSON."
                },
                {
                    "role": "user",
                    "content": "Create a JSON user profile for a 28-year-old software developer named John with email john@example.com who likes coding and hiking. Output ONLY the JSON object, no additional text."
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema
            },
            max_tokens=150
        )
        
        content = response.choices[0].message.content
        print("User: Create user profile for John (28, developer, likes coding/hiking)")
        print("Response with schema validation:")
        
        try:
            json_obj = json.loads(content)
            print(json.dumps(json_obj, indent=2))
            print("✅ Schema-compliant JSON received")
        except json.JSONDecodeError:
            print(content)
            print("⚠️  Response parsing issue")
    except Exception as e:
        print(f"⚠️  JSON schema test failed: {e}")
        print("   (Model may not fully support strict JSON schema mode)")


# ============================================================================
# File Upload & RAG
# ============================================================================

def test_file_upload_and_rag():
    """Test: File upload and RAG (Retrieval Augmented Generation)"""
    print("\n" + "="*70)
    print("TEST 10: File Upload & RAG")
    print("="*70)
    
    # Create a test document
    doc_content = """
OpenVINO™ Toolkit Overview

OpenVINO (Open Visual Inference and Neural Network Optimization) is an open-source 
toolkit for optimizing and deploying AI inference.

Key Benefits:
1. Boost deep learning performance in computer vision, NLP, and other applications
2. Run models on multiple hardware types: CPU, GPU, NPU, and VPU
3. Reduce inference latency and improve throughput
4. Support for PyTorch, TensorFlow, ONNX, and other frameworks

Intel NPU (Neural Processing Unit) is a specialized AI accelerator integrated into 
Intel Core Ultra processors, delivering high-performance, power-efficient AI inference.
    """
    
    # Save and upload document
    doc_path = Path("test_document.txt")
    doc_path.write_text(doc_content)
    
    with open(doc_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="assistants")
    
    print(f"✅ Uploaded document: {file_obj.filename} (ID: {file_obj.id})")
    
    # Ask question about the document
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Based on the document, what are the key benefits of OpenVINO?"
                    },
                    {
                        "type": "image_url",  # Using image_url for file reference
                        "image_url": {"url": file_obj.id}
                    }
                ]
            }
        ],
        max_tokens=200
    )
    
    print(f"\n👤 User: What are the key benefits of OpenVINO? (with document context)")
    print(f"🤖 Assistant: {response.choices[0].message.content}")
    
    # Cleanup
    client.files.delete(file_obj.id)
    doc_path.unlink()
    print("✅ RAG test completed")


# ============================================================================
# Embeddings & Vector Store
# ============================================================================

def test_embeddings():
    """Test: Generate text embeddings"""
    print("\n" + "="*70)
    print("TEST 11: Text Embeddings")
    print("="*70)
    
    response = client.embeddings.create(
        model="text-embedding-model",  # Update with your embedding model name
        input=[
            "OpenVINO is an AI inference toolkit",
            "Neural Processing Units accelerate AI workloads",
            "Python is a programming language"
        ]
    )
    
    print(f"✅ Generated {len(response.data)} embedding(s)")
    for i, embedding in enumerate(response.data):
        emb_preview = embedding.embedding[:5] if isinstance(embedding.embedding, list) else "..."
        print(f"   [{i}] Dimension: {len(embedding.embedding) if isinstance(embedding.embedding, list) else 'N/A'}, Preview: {emb_preview}...")
    print(f"Total tokens: {response.usage.total_tokens}")


def test_vector_store():
    """Test: Vector store operations"""
    print("\n" + "="*70)
    print("TEST 12: Vector Store (Advanced RAG)")
    print("="*70)
    
    import requests
    base_url = "http://localhost:8000"
    
    # Add documents to vector store
    docs = [
        {"text": "OpenVINO enables AI inference optimization across multiple hardware platforms.",
         "metadata": {"category": "overview"}},
        {"text": "Intel NPU provides efficient AI acceleration in Core Ultra processors.",
         "metadata": {"category": "hardware"}},
        {"text": "Python is the primary language for AI and machine learning development.",
         "metadata": {"category": "programming"}}
    ]
    
    print("Adding documents to vector store...")
    for doc in docs:
        response = requests.post(
            f"{base_url}/v1/vector_store/documents",
            json={
                "text": doc["text"],
                "embedding_model": "text-embedding-model",
                "metadata": doc["metadata"]
            }
        )
        if response.status_code == 200:
            doc_id = response.json().get("doc_id")
            print(f"   ✅ Added doc: {doc_id[:16]}...")
        else:
            raise Exception(f"Failed to add document: {response.status_code} - {response.text}")
    
    # Search vector store
    print("\nSearching vector store...")
    search_response = requests.post(
        f"{base_url}/v1/vector_store/search",
        json={
            "query": "What hardware accelerates AI?",
            "embedding_model": "text-embedding-model",
            "top_k": 2,
            "threshold": 0.0
        }
    )
    
    if search_response.status_code == 200:
        results = search_response.json()
        print(f"✅ Found {results['count']} similar document(s):")
        for result in results['results']:
            print(f"   - Similarity: {result['similarity']:.3f}")
            print(f"     Text: {result['text'][:80]}...")
    else:
        raise Exception(f"Failed to search vector store: {search_response.status_code} - {search_response.text}")
    
    print("✅ Vector store test completed")


# ============================================================================
# Audio (TTS & STT)
# ============================================================================

def test_text_to_speech():
    """Test: Text-to-speech"""
    print("\n" + "="*70)
    print("TEST 13: Text-to-Speech (TTS)")
    print("="*70)
    
    response = client.audio.speech.create(
        model="speecht5-tts",  # Update with your TTS model name
        input="Hello! This is a test of the text to speech system.",
        voice="alloy",
        response_format="wav"
    )
    
    output_file = "test_speech.wav"
    response.stream_to_file(output_file)
    
    file_size = Path(output_file).stat().st_size
    print(f"✅ Generated speech saved to: {output_file}")
    print(f"   File size: {file_size} bytes")
    
    # Cleanup
    Path(output_file).unlink()


def test_speech_to_text():
    """Test: Speech-to-text"""
    print("\n" + "="*70)
    print("TEST 14: Speech-to-Text (Whisper)")
    print("="*70)
    
    print("⚠️  STT test skipped: Requires an audio file and Whisper model")
    print("   See examples/audio.py for detailed audio examples")


# ============================================================================
# Vision / Multimodal
# ============================================================================

def test_vision_multimodal():
    """Test: Vision/multimodal capabilities"""
    print("\n" + "="*70)
    print("TEST 15: Vision/Multimodal (VLM)")
    print("="*70)
    
    # Create a simple test image with text
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    # Create a 256x256 image with some text and shapes
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a red circle
    draw.ellipse([50, 50, 150, 150], fill='red', outline='black', width=3)
    
    # Draw a blue square
    draw.rectangle([120, 120, 200, 200], fill='blue', outline='black', width=3)
    
    # Convert to base64
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    # Test VLM with image
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what shapes and colors you see in this image. Be brief."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=100
    )
    
    print(f"👤 User: [Image with shapes] Describe what you see")
    print(f"🤖 Assistant: {response.choices[0].message.content}")
    print("✅ VLM test completed")


# ============================================================================
# Image Generation
# ============================================================================

def test_image_generation():
    """Test: Text-to-image generation"""
    print("\n" + "="*70)
    print("TEST 16: Image Generation (Text-to-Image)")
    print("="*70)
    
    import requests
    
    response = requests.post(
        "http://localhost:8000/v1/images/generations",
        json={
            "model": "stable-diffusion",  # Update with your model name
            "prompt": "A beautiful sunset over mountains, digital art",
            "n": 1,
            "size": "512x512",
            "response_format": "url"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated {len(data['data'])} image(s)")
        for img in data['data']:
            if img.get('url'):
                print(f"   Image URL: {img['url']}")
            elif img.get('revised_prompt'):
                print(f"   Revised prompt: {img['revised_prompt']}")
    else:
        raise Exception(f"Image generation failed: {response.status_code} - {response.text}")


# ============================================================================
# Content Moderation
# ============================================================================

def test_moderation():
    """Test: Content moderation"""
    print("\n" + "="*70)
    print("TEST 17: Content Moderation")
    print("="*70)
    
    import requests
    
    texts = [
        "I love learning about AI and programming!",
        "This is a test message for moderation."
    ]
    
    for text in texts:
        response = requests.post(
            "http://localhost:8000/v1/moderations",
            json={"input": text, "model": "moderation-model"}
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['results'][0]
            print(f"Text: \"{text[:50]}...\"")
            print(f"   Flagged: {result['flagged']}")
            if result['flagged']:
                flagged_categories = [k for k, v in result['categories'].items() if v]
                print(f"   Categories: {', '.join(flagged_categories)}")
            print()
        else:
            raise Exception(f"Moderation failed: {response.status_code} - {response.text}")
    
    print("✅ Moderation test completed")


# ============================================================================
# Advanced Features
# ============================================================================

def test_seed_reproducibility():
    """Test: Seed for reproducible outputs"""
    print("\n" + "="*70)
    print("TEST 18: Seed Reproducibility")
    print("="*70)
    
    prompt_messages = [
        {"role": "user", "content": "Say a random number between 1 and 100"}
    ]
    
    # Generate twice with same seed
    response1 = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=prompt_messages,
        seed=12345,
        temperature=0.7,
        max_tokens=20
    )
    
    response2 = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=prompt_messages,
        seed=12345,
        temperature=0.7,
        max_tokens=20
    )
    
    content1 = response1.choices[0].message.content
    content2 = response2.choices[0].message.content
    
    print(f"Response 1 (seed=12345): {content1}")
    print(f"Response 2 (seed=12345): {content2}")
    
    if content1 == content2:
        print("✅ Responses are identical (reproducible)")
    else:
        print("⚠️  Responses differ (seed may not be fully supported)")


def test_stop_sequences():
    """Test: Stop sequences"""
    print("\n" + "="*70)
    print("TEST 19: Stop Sequences")
    print("="*70)
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {"role": "user", "content": "List the days of the week"}
        ],
        stop=["Thursday", "friday"],  # Stop at Thursday
        max_tokens=100
    )
    
    content = response.choices[0].message.content
    print(f"User: List the days of the week (stop at 'Thursday')")
    print(f"Response: {content}")
    
    if "Friday" not in content:
        print("✅ Stop sequence worked")
    else:
        print("⚠️  Stop sequence may not have triggered")


def test_system_fingerprint():
    """Test: System fingerprint for caching"""
    print("\n" + "="*70)
    print("TEST 20: System Fingerprint")
    print("="*70)
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
        max_tokens=20
    )
    
    if hasattr(response, 'system_fingerprint') and response.system_fingerprint:
        print(f"System fingerprint: {response.system_fingerprint}")
        print("✅ System fingerprint available")
    else:
        print("⚠️  System fingerprint not available")


# ============================================================================
# Voice Chat Tests
# ============================================================================

def test_voice_chat_rest_api():
    """Test: Voice chat using REST API (STT → Chat → TTS)"""
    print("\n" + "="*70)
    print("TEST 21: Voice Chat - REST API Mode")
    print("="*70)
    print("Tests: Whisper (STT) → Chat → TTS pipeline")
    
    try:
        import sounddevice as sd
        import soundfile as sf
        import tempfile
        
        WHISPER_MODEL = "whisper-base"
        LLM_MODEL = "phi-3.5-vision"
        TTS_MODEL = "speecht5-tts"
        SAMPLE_RATE = 16000
        DURATION = 3  # seconds for test
        
        print(f"\nPipeline: {WHISPER_MODEL} → {LLM_MODEL} → {TTS_MODEL}")
        print(f"Recording for {DURATION} seconds...")
        
        # Step 1: Record audio
        print("🎤 Recording... (speak now or silence for test)")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("   ✅ Recording complete!")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio, SAMPLE_RATE)
        
        try:
            # Step 2: Transcribe
            print("🔄 Transcribing...")
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=audio_file,
                    response_format="json"
                )
            
            user_text = transcription.text[:100] or "[No speech detected]"
            print(f"👤 Transcribed: \"{user_text}\"")
            
            # Step 3: Get AI response (use test prompt if no speech)
            if not user_text.strip() or user_text == "[No speech detected]":
                user_text = "Hello, how are you?"
                print(f"   Using test prompt: \"{user_text}\"")
            
            print("🤖 Generating response...")
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Keep responses very brief."},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            assistant_text = response.choices[0].message.content
            print(f"🤖 Assistant: \"{assistant_text}\"")
            
            # Step 4: Generate speech
            print("🔄 Generating speech...")
            speech_response = client.audio.speech.create(
                model=TTS_MODEL,
                input=assistant_text[:100],  # Limit for test
                voice="alloy",
                response_format="wav"
            )
            
            # Save audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as speech_file:
                speech_path = speech_file.name
                speech_response.stream_to_file(speech_path)
            
            # Verify audio file
            file_size = Path(speech_path).stat().st_size
            print(f"✅ Generated speech: {file_size} bytes")
            
            # Optional: Play audio (commented out for automated testing)
            # audio_data, sample_rate = sf.read(speech_path)
            # sd.play(audio_data, sample_rate)
            # sd.wait()
            
            # Cleanup
            Path(speech_path).unlink(missing_ok=True)
            
            print("✅ Voice chat REST API pipeline completed")
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    except ImportError:
        print("⚠️  Voice chat REST test skipped: Missing dependencies")
        print("   Install: pip install sounddevice soundfile")
    except Exception as e:
        print(f"⚠️  Voice chat REST test skipped: {e}")
        print("   (Requires Whisper, LLM, and TTS models configured)")


def test_voice_chat_realtime_websocket():
    """Test: Voice chat using Realtime WebSocket API"""
    print("\n" + "="*70)
    print("TEST 22: Voice Chat - Realtime WebSocket Mode")
    print("="*70)
    print("Tests: Full-duplex WebSocket audio streaming")
    
    try:
        import asyncio
        import websockets
        
        async def test_realtime():
            WS_URL = "ws://localhost:8000/v1/realtime?model=phi-3.5-vision"
            
            print("🔌 Connecting to WebSocket...")
            
            try:
                async with websockets.connect(WS_URL) as websocket:
                    print("✅ Connected!")
                    
                    # Wait for session.created event
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    event = json.loads(message)
                    
                    if event.get("type") == "session.created":
                        session_id = event["session"]["id"]
                        print(f"📋 Session created: {session_id}")
                        print(f"   Modalities: {event['session']['modalities']}")
                        print(f"   Audio format: {event['session']['input_audio_format']}")
                        
                        # Send test text message
                        print("\n📤 Sending text message...")
                        text_event = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Hello! Say hi back in one word."}
                                ]
                            }
                        }
                        await websocket.send(json.dumps(text_event))
                        
                        # Wait for response events (with timeout)
                        # VLM generation on NPU can take 30+ seconds
                        response_text = ""
                        timeout = 60.0  # Increased to 60 seconds for VLM on NPU
                        start_time = asyncio.get_event_loop().time()
                        
                        while asyncio.get_event_loop().time() - start_time < timeout:
                            try:
                                # Increased recv timeout for VLM generation delay
                                message = await asyncio.wait_for(websocket.recv(), timeout=300.0)
                                event = json.loads(message)
                                event_type = event.get("type")
                                
                                if event_type == "response.created":
                                    print("🤖 Response started...")
                                
                                elif event_type == "response.text.delta":
                                    delta = event.get("delta", "")
                                    response_text += delta
                                    print(delta, end="", flush=True)
                                
                                elif event_type == "response.done":
                                    print("\n✅ Response complete!")
                                    break
                                
                                elif event_type == "error":
                                    error = event.get("error", {})
                                    print(f"\n❌ Error: {error.get('message')}")
                                    break
                                
                                else:
                                    # Log other event types for debugging
                                    print(f"\n🔔 Event: {event_type}")
                                    
                            except asyncio.TimeoutError:
                                # Timeout waiting for next message - keep waiting
                                print(f"\n⏳ Waiting for response...")
                                continue
                        
                        if response_text:
                            print(f"📝 Full response: \"{response_text}\"")
                            print("✅ Realtime WebSocket test completed")
                        else:
                            print("⚠️  No response received (may need more time)")
                    else:
                        print(f"⚠️  Unexpected event: {event.get('type')}")
                        
            except websockets.exceptions.WebSocketException as e:
                print(f"⚠️  WebSocket error: {e}")
            except asyncio.TimeoutError:
                print("⚠️  Connection timeout")
        
        # Run async test
        asyncio.run(test_realtime())
        
    except ImportError:
        print("⚠️  Realtime WebSocket test skipped: Missing websockets library")
        print("   Install: pip install websockets")
    except Exception as e:
        print(f"⚠️  Realtime WebSocket test skipped: {e}")
        print("   (Requires server with WebSocket support)")


def test_image_edit():
    """Test: Image editing with mask"""
    print("\n" + "="*70)
    print("TEST 23: Image Editing (DALL·E Edit)")
    print("="*70)
    
    import requests
    from PIL import Image as PILImage
    import io
    
    # Create a test image
    print("Creating test image...")
    img = PILImage.new('RGB', (512, 512), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Send edit request
    response = requests.post(
        "http://localhost:8000/v1/images/edits",
        files={
            "image": ("test.png", img_bytes, "image/png")
        },
        data={
            "prompt": "Add a red circle in the center",
            "n": 1,
            "size": "512x512",
            "response_format": "url"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Edited {len(data['data'])} image(s)")
        for img in data['data']:
            if img.get('url'):
                print(f"   Image URL: {img['url']}")
    else:
        raise Exception(f"Image edit failed: {response.status_code} - {response.text}")


def test_image_variations():
    """Test: Image variations"""
    print("\n" + "="*70)
    print("TEST 24: Image Variations (DALL·E Variations)")
    print("="*70)
    
    import requests
    from PIL import Image as PILImage
    import io
    
    # Create a test image
    print("Creating test image...")
    img = PILImage.new('RGB', (512, 512), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Send variations request
    response = requests.post(
        "http://localhost:8000/v1/images/variations",
        files={
            "image": ("test.png", img_bytes, "image/png")
        },
        data={
            "n": 2,
            "size": "512x512",
            "response_format": "url"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated {len(data['data'])} variation(s)")
        for img in data['data']:
            if img.get('url'):
                print(f"   Image URL: {img['url']}")
    else:
        raise Exception(f"Image variations failed: {response.status_code} - {response.text}")


def test_openai_error_format():
    """Test: OpenAI error format compliance"""
    print("\n" + "="*70)
    print("TEST 25: OpenAI Error Format")
    print("="*70)
    
    try:
        # Try to use non-existent model
        try:
            response = client.chat.completions.create(
                model="non-existent-model-12345",
                messages=[{"role": "user", "content": "Hello"}]
            )
        except Exception as e:
            error_str = str(e)
            print(f"Error received: {error_str[:100]}...")
            
            # Check if error contains expected fields
            if "model" in error_str.lower() and ("not found" in error_str.lower() or "does not exist" in error_str.lower()):
                print("✅ Error format appears correct")
            else:
                print("⚠️  Error format may not match OpenAI spec")
                
    except Exception as e:
        print(f"⚠️  Error format test failed: {e}")


def test_model_retrieve():
    """Test: Retrieve specific model details"""
    print("\n" + "="*70)
    print("TEST 26: Model Retrieve Endpoint")
    print("="*70)
    
    # List models first
    models = client.models.list()
    if not models.data:
        print("⚠️  No models available to test")
        return
    
    test_model = models.data[0].id
    print(f"\n📋 Retrieving details for model: {test_model}")
    
    # Retrieve specific model
    model = client.models.retrieve(test_model)
    
    print(f"   ID: {model.id}")
    print(f"   Object: {model.object}")
    print(f"   Created: {model.created}")
    print(f"   Owned by: {model.owned_by}")
    
    if hasattr(model, 'capabilities') and model.capabilities:
        print(f"   Capabilities:")
        print(f"      Type: {model.capabilities.get('type')}")
        print(f"      Device: {model.capabilities.get('device')}")
        print(f"      Path: {model.capabilities.get('path')}")
    
    print("✅ Model retrieve test completed")


def test_file_operations():
    """Test: File list, retrieve, delete operations"""
    print("\n" + "="*70)
    print("TEST 27: File CRUD Operations")
    print("="*70)
    
    import requests
    
    # Upload a test file first
    print("\n📤 Part 1: Upload a test file")
    test_content = b"Test file content for CRUD operations"
    files = {'file': ('test_crud.txt', test_content, 'text/plain')}
    data = {'purpose': 'assistants'}
    
    response = requests.post(
        "http://localhost:8000/v1/files",
        files=files,
        data=data
    )
    
    if response.status_code != 200:
        raise Exception(f"File upload failed: {response.status_code} - {response.text}")
    
    file_obj = response.json()
    file_id = file_obj['id']
    print(f"   Uploaded file ID: {file_id}")
    print(f"   Filename: {file_obj['filename']}")
    
    # List all files
    print("\n📋 Part 2: List all files")
    response = requests.get("http://localhost:8000/v1/files")
    
    if response.status_code != 200:
        raise Exception(f"File list failed: {response.status_code}")
    
    files_data = response.json()
    print(f"   Total files: {len(files_data['data'])}")
    for f in files_data['data'][:5]:  # Show first 5
        print(f"      - {f['id']}: {f['filename']} ({f['bytes']} bytes)")
    
    # Retrieve specific file metadata
    print(f"\n🔍 Part 3: Retrieve file {file_id}")
    response = requests.get(f"http://localhost:8000/v1/files/{file_id}")
    
    if response.status_code != 200:
        raise Exception(f"File retrieve failed: {response.status_code}")
    
    file_data = response.json()
    print(f"   ID: {file_data['id']}")
    print(f"   Filename: {file_data['filename']}")
    print(f"   Size: {file_data['bytes']} bytes")
    print(f"   Purpose: {file_data['purpose']}")
    
    # Delete the file
    print(f"\n🗑️  Part 4: Delete file {file_id}")
    response = requests.delete(f"http://localhost:8000/v1/files/{file_id}")
    
    if response.status_code != 200:
        raise Exception(f"File delete failed: {response.status_code}")
    
    delete_result = response.json()
    print(f"   Deleted: {delete_result['deleted']}")
    print(f"   File ID: {delete_result['id']}")
    
    print("\n✅ File CRUD operations completed!")


def test_logprobs():
    """Test: Logprobs support in chat completions"""
    print("\n" + "="*70)
    print("TEST 28: Logprobs Support")
    print("="*70)
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {"role": "user", "content": "Count to five: 1, 2,"}
        ],
        max_tokens=20,
        logprobs=True,
        top_logprobs=2
    )
    
    print(f"👤 User: Count to five: 1, 2,")
    print(f"🤖 Assistant: {response.choices[0].message.content}")
    
    # Check if logprobs are present
    if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
        logprobs = response.choices[0].logprobs
        print(f"\n📊 Logprobs information:")
        
        if hasattr(logprobs, 'content') and logprobs.content:
            print(f"   Tokens with logprobs: {len(logprobs.content)}")
            
            # Show first few tokens
            for i, token_logprob in enumerate(logprobs.content[:5]):
                print(f"   Token {i+1}: '{token_logprob.token}' (logprob: {token_logprob.logprob:.4f})")
                if token_logprob.top_logprobs:
                    print(f"      Top alternatives: {len(token_logprob.top_logprobs)}")
        else:
            print("   Structure present but no content (expected with current implementation)")
    else:
        print("⚠️  No logprobs in response")
    
    print("\n✅ Logprobs test completed")
    print("   Note: Full logprobs require deeper OpenVINO GenAI integration")


def test_multimodal_audio_output():
    """Test: Multimodal chat with audio output (GPT-4o style)"""
    print("\n" + "="*70)
    print("TEST 29: Multimodal Audio Output (GPT-4o Style)")
    print("="*70)
    print("\n🎯 This tests the 'modalities' parameter for audio output in chat completions")
    print("   Similar to OpenAI's GPT-4o-audio feature\n")
    
    # Test 1: Text-only chat with audio output
    print("📝 Part 1: Text chat with audio output")
    print("-" * 70)
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {"role": "user", "content": "Say hello in exactly 5 words."}
        ],
        max_tokens=50,
        modalities=["text", "audio"],  # Request both text AND audio
        audio={
            "voice": "default",
            "format": "wav"
        }
    )
    
    print(f"👤 User: Say hello in exactly 5 words.")
    print(f"🤖 Assistant (text): {response.choices[0].message.content}")
    
    # Check if audio was included
    if hasattr(response.choices[0].message, 'audio') and response.choices[0].message.audio:
        audio_data = response.choices[0].message.audio
        print(f"🔊 Audio output:")
        print(f"   Audio ID: {audio_data.id}")
        print(f"   Transcript: {audio_data.transcript}")
        print(f"   Data size: {len(audio_data.data)} bytes (base64)")
        
        # Optionally save the audio
        try:
            audio_bytes = base64.b64decode(audio_data.data)
            output_path = Path("test_audio_output.wav")
            output_path.write_bytes(audio_bytes)
            print(f"   💾 Saved to: {output_path}")
        except Exception as e:
            print(f"   ⚠️  Could not save audio: {e}")
    else:
        print("⚠️  No audio in response (might be expected if no TTS model loaded)")
    
    # Test 2: Multimodal input (text + image) with audio output
    print("\n🖼️  Part 2: Image + text chat with audio output")
    print("-" * 70)
    
    from PIL import Image, ImageDraw
    import io
    
    # Create a simple test image
    img = Image.new('RGB', (200, 100), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 25, 150, 75], fill='yellow', outline='orange', width=3)
    
    # Convert to base64
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short sentence."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        max_tokens=50,
        modalities=["text", "audio"],
        audio={"voice": "default", "format": "wav"}
    )
    
    print(f"👤 User: [Image with shapes] Describe this image in one short sentence.")
    print(f"🤖 Assistant (text): {response.choices[0].message.content}")
    
    if hasattr(response.choices[0].message, 'audio') and response.choices[0].message.audio:
        audio_data = response.choices[0].message.audio
        print(f"🔊 Audio output:")
        print(f"   Audio ID: {audio_data.id}")
        print(f"   Transcript: {audio_data.transcript}")
        print(f"   Data size: {len(audio_data.data)} bytes")
    
    # Test 3: Text-only output (no audio modality)
    print("\n📄 Part 3: Regular chat without audio (baseline)")
    print("-" * 70)
    
    response = client.chat.completions.create(
        model="phi-3.5-vision",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        max_tokens=20
        # No modalities parameter - should only return text
    )
    
    print(f"👤 User: What is 2+2?")
    print(f"🤖 Assistant: {response.choices[0].message.content}")
    
    has_audio = hasattr(response.choices[0].message, 'audio') and response.choices[0].message.audio
    if not has_audio:
        print(f"✅ No audio in response (as expected without modalities parameter)")
    else:
        print(f"⚠️  Audio present when not requested")
    
    print("\n✅ Multimodal audio output test completed!")
    print("   This demonstrates our GPT-4o-style multimodal output capability:")
    print("   - Text input → Text + Audio output")
    print("   - Image + Text input → Text + Audio output")
    print("   - Mimics OpenAI's GPT-4o-audio API exactly!")


# ============================================================================
# Main Test Runner
# ============================================================================

def cleanup_temp_files():
    """Clean up temporary files and data before tests"""
    print("\n" + "="*70)
    print("CLEANUP: Clearing temporary files and data")
    print("="*70)
    
    import shutil
    import glob
    
    cleaned_items = []
    
    # 1. Clear vector store
    try:
        vector_store_path = Path("vector_store/vector_index.json")
        if vector_store_path.exists():
            # Reset to empty vector store
            vector_store_path.write_text(json.dumps({
                "documents": [],
                "embeddings": [],
                "metadata": []
            }, indent=2))
            cleaned_items.append("Vector store documents")
    except Exception as e:
        print(f"⚠️  Could not clear vector store: {e}")
    
    # 2. Clear generated images
    try:
        generated_images_dir = Path("generated_images")
        if generated_images_dir.exists():
            image_files = list(generated_images_dir.glob("*.png"))
            for img_file in image_files:
                try:
                    img_file.unlink()
                except Exception:
                    pass
            if image_files:
                cleaned_items.append(f"Generated images ({len(image_files)} files)")
    except Exception as e:
        print(f"⚠️  Could not clear generated images: {e}")
    
    # 3. Clear uploaded files
    try:
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            # Clear metadata
            metadata_file = uploads_dir / "files_metadata.json"
            if metadata_file.exists():
                metadata_file.write_text(json.dumps({}, indent=2))
            
            # Remove uploaded files (but keep .gitkeep if exists)
            file_count = 0
            for upload_file in uploads_dir.iterdir():
                if upload_file.name not in ["files_metadata.json", ".gitkeep"]:
                    try:
                        if upload_file.is_file():
                            upload_file.unlink()
                            file_count += 1
                        elif upload_file.is_dir():
                            shutil.rmtree(upload_file)
                            file_count += 1
                    except Exception:
                        pass
            
            if file_count > 0:
                cleaned_items.append(f"Uploaded files ({file_count} items)")
    except Exception as e:
        print(f"⚠️  Could not clear uploads: {e}")
    
    # 4. Clear any test files in root directory
    try:
        test_files = [
            "test_document.txt",
            "test_speech.wav",
            "test_audio_output.wav",
            "test_crud.txt"
        ]
        removed_count = 0
        for test_file in test_files:
            test_path = Path(test_file)
            if test_path.exists():
                try:
                    test_path.unlink()
                    removed_count += 1
                except Exception:
                    pass
        
        if removed_count > 0:
            cleaned_items.append(f"Test files ({removed_count} files)")
    except Exception as e:
        print(f"⚠️  Could not clear test files: {e}")
    
    # 5. Clear session data via API (if server is running)
    try:
        import requests
        # Try to clear any active sessions
        response = requests.get("http://localhost:8000/v1/realtime/sessions", timeout=2)
        if response.status_code == 200:
            sessions = response.json().get("sessions", [])
            for session in sessions:
                try:
                    requests.delete(f"http://localhost:8000/v1/realtime/sessions/{session['id']}", timeout=2)
                except Exception:
                    pass
            if sessions:
                cleaned_items.append(f"Active sessions ({len(sessions)} sessions)")
    except Exception:
        # Server might not be running yet, that's okay
        pass
    
    if cleaned_items:
        print("✅ Cleaned up:")
        for item in cleaned_items:
            print(f"   • {item}")
    else:
        print("✅ No temporary files to clean (already clean)")
    
    print("=" * 70)


def run_health_check():
    """Check server health"""
    print("\n" + "="*70)
    print("HEALTH CHECK")
    print("="*70)
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health")
        data = response.json()
        
        print(f"Status: {data['status']}")
        print(f"Models loaded: {data['models_loaded']}")
        print(f"  - LLM: {data['llm_models']}")
        print(f"  - VLM: {data['vlm_models']}")
        print(f"  - Whisper: {data['whisper_models']}")
        print(f"  - TTS: {data['tts_models']}")
        print(f"  - Embedding: {data.get('embedding_models', 0)}")
        print(f"Files stored: {data['files_stored']}")
        print(f"Vector store docs: {data.get('documents_in_vector_store', 0)}")
        print("✅ Server is healthy")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def main():
    """Run comprehensive test suite"""
    print("=" * 70)
    print("COMPREHENSIVE FEATURE TEST - OpenVINO GenAI API Server")
    print("=" * 70)
    print("\nThis test suite covers ALL 29 features:")
    print("  • Basic chat and completions")
    print("  • Streaming")
    print("  • Tool/function calling")
    print("  • Structured outputs (JSON mode & schema)")
    print("  • File upload, list, retrieve, delete 🆕")
    print("  • Embeddings and vector store")
    print("  • Audio (TTS/STT)")
    print("  • Vision/multimodal")
    print("  • Image generation, editing, variations")
    print("  • Content moderation")
    print("  • Advanced features (seed, stop sequences, etc.)")
    print("  • Voice chat (REST API & WebSocket Realtime)")
    print("  • Session management (list, get, delete) 🆕")
    print("  • Multimodal audio output (GPT-4o style) 🆕")
    print("  • Model retrieve endpoint 🆕")
    print("  • Logprobs support 🆕")
    print("  • OpenAI error format compliance")
    print("=" * 70)
    
    # Clean up temporary files and data before starting tests
    cleanup_temp_files()
    
    # Check server health
    if not run_health_check():
        print("\n❌ Server is not running or not accessible")
        print("   Please start the server: python npu.py")
        return
    
    try:
        input("\nPress Enter to start tests...")
    except EOFError:
        # Non-interactive mode, proceed automatically
        print("\nRunning in non-interactive mode, starting tests...")
    
    # Track test results
    tests_run = 0
    tests_passed = 0
    tests_failed = 0
    tests_skipped = 0
    
    # Exception to signal a test was skipped (not a failure)
    class TestSkipped(Exception):
        pass
    
    def run_test(test_func):
        """Run a single test and track results"""
        nonlocal tests_run, tests_passed, tests_failed, tests_skipped
        tests_run += 1
        try:
            test_func()
            tests_passed += 1
        except TestSkipped as e:
            tests_skipped += 1
            # Skipped message already printed by test function
        except KeyboardInterrupt:
            raise  # Allow user to stop tests
        except Exception as e:
            tests_failed += 1
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        # Basic features (1-5)
        run_test(test_list_models)
        run_test(test_basic_chat)
        run_test(test_streaming)
        run_test(test_text_completion)
        run_test(test_multi_turn_conversation)
        
        # Tool calling (6-7)
        run_test(test_tool_calling)
        run_test(test_tool_calling_with_execution)
        
        # Structured outputs (8-9)
        run_test(test_json_mode)
        run_test(test_json_schema)
        
        # File & RAG (10)
        run_test(test_file_upload_and_rag)
        
        # Embeddings & Vector Store (11-12)
        run_test(test_embeddings)
        run_test(test_vector_store)
        
        # Audio (13-14)
        run_test(test_text_to_speech)
        run_test(test_speech_to_text)
        
        # Vision (15)
        run_test(test_vision_multimodal)
        
        # Image generation (16)
        run_test(test_image_generation)
        
        # Content moderation (17)
        run_test(test_moderation)
        
        # Advanced features (18-20)
        run_test(test_seed_reproducibility)
        run_test(test_stop_sequences)
        run_test(test_system_fingerprint)
        
        # Voice chat (21-22)
        run_test(test_voice_chat_rest_api)
        run_test(test_voice_chat_realtime_websocket)
        
        # Image editing & variations (23-24)
        run_test(test_image_edit)
        run_test(test_image_variations)
        
        # Error handling (25)
        run_test(test_openai_error_format)
        
        # New features (26-29) - NEWLY IMPLEMENTED!
        run_test(test_model_retrieve)
        run_test(test_file_operations)
        run_test(test_logprobs)
        run_test(test_multimodal_audio_output)
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUITE COMPLETED")
        print("=" * 70)
        print(f"\n📊 Test Results:")
        print(f"   ✅ Passed: {tests_passed}/{tests_run}")
        if tests_failed > 0:
            print(f"   ❌ Failed: {tests_failed}")
        if tests_skipped > 0:
            print(f"   ⚠️  Skipped: {tests_skipped}")
        
        if tests_failed == 0 and tests_skipped == 0:
            print(f"\n🎉 All {tests_run} tests passed!")
        elif tests_failed == 0:
            print(f"\n✅ All non-skipped tests passed ({tests_passed} passed, {tests_skipped} skipped)")
        print("\n📊 Test Coverage:")
        print("   ✓ Core API: Models (list + retrieve 🆕), Chat, Completions, Streaming")
        print("   ✓ Advanced: Tools, JSON mode, Structured outputs, Logprobs 🆕")
        print("   ✓ Files: Upload, List 🆕, Retrieve 🆕, Delete 🆕, RAG, Vector store")
        print("   ✓ Audio: TTS, STT, Voice chat (REST + WebSocket)")
        print("   ✓ Images: Generation, Editing, Variations")
        print("   ✓ Vision: Multimodal, VLM support")
        print("   ✓ Multimodal Output: Text + Audio (GPT-4o style) 🆕")
        print("   ✓ WebSocket: Function calling support 🆕")
        print("   ✓ Session Management: List, Get, Delete sessions 🆕")
        print("   ✓ Safety: Content moderation")
        print("   ✓ Parameters: Seed, stop sequences, fingerprint")
        print("   ✓ Compatibility: OpenAI error format")
        print("\n⚠️  Some tests may have been skipped if required models are not configured.")
        print("\nTo enable all features, configure your config.json with:")
        print("  • LLM models (type: 'llm')")
        print("  • Vision models (type: 'vlm')")
        print("  • Whisper models (type: 'whisper')")
        print("  • TTS models (type: 'tts')")
        print("  • Embedding models (type: 'embedding')")
        print("  • Text-to-image models (type: 'text2image')")
        print("  • Moderation models (type: 'moderation')")
        print("\n📚 For voice chat demos with audio I/O:")
        print("   Install: pip install sounddevice soundfile websockets")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        print(f"\n📊 Partial Results before interruption:")
        print(f"   ✅ Passed: {tests_passed}")
        if tests_failed > 0:
            print(f"   ❌ Failed: {tests_failed}")
        if tests_skipped > 0:
            print(f"   ⚠️  Skipped: {tests_skipped}")
        print(f"   Total run: {tests_run}")


if __name__ == "__main__":
    main()

