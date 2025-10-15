"""
Audio Examples - OpenVINO GenAI API
Demonstrates speech-to-text (Whisper) and text-to-speech capabilities
"""

from openai import OpenAI
from pathlib import Path

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)


def example_text_to_speech():
    """Example: Generate speech from text"""
    print("=" * 60)
    print("Example 1: Text-to-Speech")
    print("=" * 60)
    
    text = "Hello! This is a test of the text to speech system using OpenVINO GenAI."
    
    print(f"Input text: {text}")
    print("Generating speech...")
    
    # Generate speech
    response = client.audio.speech.create(
        model="speecht5-tts",  # Replace with your TTS model name
        input=text,
        voice="alloy",
        response_format="wav"
    )
    
    # Save to file
    output_file = "output_speech.wav"
    response.stream_to_file(output_file)
    
    print(f"✅ Speech generated and saved to: {output_file}")
    print()


def example_speech_to_text():
    """Example: Transcribe audio to text"""
    print("=" * 60)
    print("Example 2: Speech-to-Text (Transcription)")
    print("=" * 60)
    
    # Note: You need an actual audio file for this
    audio_file = "sample_audio.wav"
    
    if not Path(audio_file).exists():
        print(f"⚠️  Audio file '{audio_file}' not found.")
        print("   Create a sample audio file or use the TTS example first.")
        print()
        return
    
    print(f"Transcribing audio file: {audio_file}")
    
    # Transcribe audio
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-base",  # Replace with your Whisper model name
            file=f,
            response_format="json"
        )
    
    print(f"Transcription: {transcription.text}")
    print()


def example_transcription_verbose():
    """Example: Transcription with verbose output"""
    print("=" * 60)
    print("Example 3: Transcription with Verbose Output")
    print("=" * 60)
    
    audio_file = "sample_audio.wav"
    
    if not Path(audio_file).exists():
        print(f"⚠️  Audio file '{audio_file}' not found.")
        print()
        return
    
    # Transcribe with verbose output
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-base",
            file=f,
            response_format="verbose_json",
            language="en"
        )
    
    print(f"Text: {transcription.text}")
    if hasattr(transcription, 'language'):
        print(f"Language: {transcription.language}")
    if hasattr(transcription, 'duration'):
        print(f"Duration: {transcription.duration}s")
    print()


def example_tts_different_formats():
    """Example: Generate speech in different formats"""
    print("=" * 60)
    print("Example 4: TTS with Different Formats")
    print("=" * 60)
    
    text = "Testing different audio formats."
    formats = ["wav", "mp3", "flac"]
    
    for fmt in formats:
        try:
            print(f"Generating {fmt.upper()} format...")
            
            response = client.audio.speech.create(
                model="speecht5-tts",
                input=text,
                voice="alloy",
                response_format=fmt
            )
            
            output_file = f"output_speech.{fmt}"
            response.stream_to_file(output_file)
            
            print(f"  ✅ Saved to: {output_file}")
            
        except Exception as e:
            print(f"  ⚠️  {fmt.upper()} format failed: {e}")
    
    print()


def example_audio_chat_pipeline():
    """Example: Complete audio pipeline - TTS + Transcription"""
    print("=" * 60)
    print("Example 5: Complete Audio Pipeline")
    print("=" * 60)
    
    # Step 1: Generate speech from text
    original_text = "The quick brown fox jumps over the lazy dog."
    print(f"1. Original text: {original_text}")
    
    print("2. Generating speech...")
    tts_response = client.audio.speech.create(
        model="speecht5-tts",
        input=original_text,
        response_format="wav"
    )
    
    temp_audio = "temp_audio.wav"
    tts_response.stream_to_file(temp_audio)
    print(f"   ✅ Speech saved to: {temp_audio}")
    
    # Step 2: Transcribe the generated audio back to text
    print("3. Transcribing speech back to text...")
    with open(temp_audio, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-base",
            file=f
        )
    
    print(f"   ✅ Transcribed text: {transcription.text}")
    
    # Compare
    print()
    print("4. Comparison:")
    print(f"   Original:    {original_text}")
    print(f"   Transcribed: {transcription.text}")
    print()


def example_multimodal_audio_conversation():
    """Example: Conversation with audio transcription"""
    print("=" * 60)
    print("Example 6: Multimodal Conversation with Audio")
    print("=" * 60)
    
    # Step 1: Create a voice note
    question = "What is machine learning?"
    
    print(f"1. Creating voice note: '{question}'")
    response = client.audio.speech.create(
        model="speecht5-tts",
        input=question,
        response_format="wav"
    )
    
    voice_note = "voice_question.wav"
    response.stream_to_file(voice_note)
    print(f"   ✅ Voice note saved")
    
    # Step 2: Transcribe the voice note
    print("2. Transcribing voice note...")
    with open(voice_note, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-base",
            file=f
        )
    
    print(f"   📝 Transcribed: {transcription.text}")
    
    # Step 3: Use transcription in a chat
    print("3. Sending to LLM...")
    chat_response = client.chat.completions.create(
        model="qwen2.5-3b",
        messages=[
            {
                "role": "user",
                "content": transcription.text
            }
        ],
        max_tokens=200
    )
    
    answer = chat_response.choices[0].message.content
    print(f"   🤖 LLM response: {answer}")
    
    # Step 4: Convert answer back to speech
    print("4. Converting answer to speech...")
    answer_audio = client.audio.speech.create(
        model="speecht5-tts",
        input=answer,
        response_format="wav"
    )
    
    answer_file = "voice_answer.wav"
    answer_audio.stream_to_file(answer_file)
    print(f"   ✅ Answer saved to: {answer_file}")
    print()


def main():
    """Run all examples"""
    print("=" * 60)
    print("Audio Examples - OpenVINO GenAI API")
    print("=" * 60)
    print()
    print("⚠️  NOTE: These examples require:")
    print("   - A Whisper model configured with type 'whisper'")
    print("   - A TTS model configured with type 'tts'")
    print("   - Update model names in the examples to match your config.json")
    print()
    
    try:
        # TTS Examples
        print("📢 TEXT-TO-SPEECH EXAMPLES")
        print()
        example_text_to_speech()
        example_tts_different_formats()
        
        # STT Examples
        print("🎤 SPEECH-TO-TEXT EXAMPLES")
        print()
        example_speech_to_text()
        example_transcription_verbose()
        
        # Combined Examples
        print("🔄 COMBINED EXAMPLES")
        print()
        example_audio_chat_pipeline()
        example_multimodal_audio_conversation()
        
        print("=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  1. The server is running: python npu.py")
        print("  2. Whisper and TTS models are configured in config.json")
        print("  3. Model names in examples match your configuration")
        print()
        print("Example config.json:")
        print("""
{
  "host": "0.0.0.0",
  "port": 8000,
  "models": [
    {
      "name": "qwen2.5-3b",
      "path": "models/Qwen/Qwen2.5-3B",
      "device": "NPU",
      "type": "llm"
    },
    {
      "name": "whisper-base",
      "path": "models/whisper-base",
      "device": "CPU",
      "type": "whisper"
    },
    {
      "name": "speecht5-tts",
      "path": "models/speecht5_tts",
      "device": "CPU",
      "type": "tts"
    }
  ]
}
        """)
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

