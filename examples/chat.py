"""
Example client for the OpenAI-compatible API server
Demonstrates various ways to interact with the server
"""

from openai import OpenAI

# Initialize client pointing to local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # API key not required but library needs something
)


def list_models():
    """List available models"""
    print("📚 Available Models:")
    models = client.models.list()
    for model in models.data:
        print(f"  - {model.id}")
    print()


def chat_completion_example():
    """Basic chat completion"""
    print("💬 Chat Completion Example:")
    
    response = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is OpenVINO?"}
        ],
        max_tokens=200,
        temperature=0.7
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.total_tokens}")
    print()


def streaming_example():
    """Streaming chat completion"""
    print("🌊 Streaming Example:")
    print("Response: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=[
            {"role": "user", "content": "Count from 1 to 10"}
        ],
        stream=True,
        max_tokens=100
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


def text_completion_example():
    """Text completion"""
    print("✏️ Text Completion Example:")
    
    response = client.completions.create(
        model="qwen2.5-3b",
        prompt="The future of artificial intelligence is",
        max_tokens=100,
        temperature=0.8
    )
    
    print(f"Response: {response.choices[0].text}")
    print()


def multi_turn_conversation():
    """Multi-turn conversation"""
    print("🗣️ Multi-turn Conversation:")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    
    # First turn
    response = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=messages,
        max_tokens=50
    )
    
    assistant_message = response.choices[0].message.content
    print(f"User: {messages[-1]['content']}")
    print(f"Assistant: {assistant_message}")
    
    # Add to conversation history
    messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": "What is it famous for?"})
    
    # Second turn
    response = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=messages,
        max_tokens=100
    )
    
    print(f"User: {messages[-1]['content']}")
    print(f"Assistant: {response.choices[0].message.content}")
    print()


def main():
    """Run all examples"""
    print("=" * 60)
    print("OpenAI-Compatible API Server - Client Examples")
    print("=" * 60)
    print()
    
    try:
        list_models()
        chat_completion_example()
        streaming_example()
        text_completion_example()
        multi_turn_conversation()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure the server is running:")
        print("  python server.py")


if __name__ == "__main__":
    main()

