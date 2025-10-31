"""
Gradio Interface for OpenVINO GenAI Server
Provides a web UI for all server features including:
- Chat & Completions (with streaming)
- Vision/Multimodal chat
- Audio transcription & TTS
- Image generation, editing, variations
- Text embeddings
- Content moderation
- File management
- Vector store (RAG)
"""

import gradio as gr
import requests
import json
import base64
import io
import tempfile
import os
from typing import List, Tuple, Optional
from pathlib import Path
import PIL.Image

# Server configuration
SERVER_URL = "http://localhost:8000"


def get_available_models():
    """Get list of available models from server"""
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
    except:
        pass
    return []


def get_model_types():
    """Get models grouped by type"""
    models = get_available_models()
    llm_models = []
    vlm_models = []
    tts_models = []
    whisper_models = []
    embedding_models = []
    image_models = []
    moderation_models = []
    
    try:
        health = requests.get(f"{SERVER_URL}/health").json()
        # We'll infer from health endpoint or config
        for model in models:
            if "qwen" in model.lower() or "vl" in model.lower():
                vlm_models.append(model)
            elif "whisper" in model.lower():
                whisper_models.append(model)
            elif "tts" in model.lower() or "speech" in model.lower():
                tts_models.append(model)
            elif "embedding" in model.lower():
                embedding_models.append(model)
            elif "image" in model.lower() or "ssd" in model.lower() or "sdxl" in model.lower():
                image_models.append(model)
            elif "moderation" in model.lower() or "toxic" in model.lower():
                moderation_models.append(model)
            else:
                llm_models.append(model)
    except:
        # Fallback: return all models for all categories
        all_models = models
        llm_models = all_models
        vlm_models = all_models
        tts_models = all_models
        whisper_models = all_models
        embedding_models = all_models
        image_models = all_models
        moderation_models = all_models
    
    return {
        "llm": llm_models if llm_models else models,
        "vlm": vlm_models if vlm_models else models,
        "tts": tts_models if tts_models else models,
        "whisper": whisper_models if whisper_models else models,
        "embedding": embedding_models if embedding_models else models,
        "image": image_models if image_models else models,
        "moderation": moderation_models if moderation_models else models
    }


# ============================================================================
# Chat & Completions
# ============================================================================

def chat_completion(messages: str, model: str, temperature: float, max_tokens: int, 
                    stream: bool, system_prompt: str):
    """Chat completion interface"""
    try:
        # Parse messages (format: user: message or system: message)
        message_list = []
        if system_prompt:
            message_list.append({"role": "system", "content": system_prompt})
        
        # Parse user messages (simple format: "user: message" or just "message")
        lines = messages.strip().split("\n")
        current_role = "user"
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("user:") or line.startswith("User:"):
                message_list.append({"role": "user", "content": line[5:].strip()})
                current_role = "user"
            elif line.startswith("assistant:") or line.startswith("Assistant:"):
                message_list.append({"role": "assistant", "content": line[10:].strip()})
                current_role = "assistant"
            elif line.startswith("system:") or line.startswith("System:"):
                message_list.append({"role": "system", "content": line[7:].strip()})
                current_role = "system"
            else:
                # Continue with current role or default to user
                if message_list and message_list[-1]["role"] == current_role:
                    message_list[-1]["content"] += "\n" + line
                else:
                    message_list.append({"role": current_role, "content": line})
        
        if not message_list or message_list[-1]["role"] != "user":
            message_list.append({"role": "user", "content": messages if messages else "Hello!"})
        
        payload = {
            "model": model,
            "messages": message_list,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            stream=stream,
            timeout=300
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_response += delta['content']
                        except:
                            pass
            return full_response
        else:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                message = data['choices'][0].get('message', {})
                content = message.get('content', '')
                
                # Note: Audio output would need special handling in Gradio
                # For now, just return text content
                return content
            return "No response generated"
            
    except requests.exceptions.RequestException as e:
        return f"Connection Error: {str(e)}\n\nMake sure the server is running at {SERVER_URL}"
    except Exception as e:
        return f"Error: {str(e)}"


def text_completion(prompt: str, model: str, temperature: float, max_tokens: int, stream: bool):
    """Text completion interface"""
    try:
        if not prompt:
            return "Please enter a prompt"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/completions",
            json=payload,
            stream=stream,
            timeout=300
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                text = data['choices'][0].get('text', '')
                                full_response += text
                        except:
                            pass
            return full_response
        else:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0].get('text', '')
            return "No response generated"
            
    except requests.exceptions.RequestException as e:
        return f"Connection Error: {str(e)}\n\nMake sure the server is running at {SERVER_URL}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Vision/Multimodal Chat
# ============================================================================

def vision_chat(message: str, image, model: str, temperature: float, max_tokens: int):
    """Vision chat with image input"""
    try:
        if image is None:
            return "Please upload an image"
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_str}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {"url": img_data_url}}
                ]
            }
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            message = data['choices'][0].get('message', {})
            return message.get('content', '')
        return "No response generated"
        
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Audio Transcription & TTS
# ============================================================================

def transcribe_audio(audio_file, model: str, language: Optional[str]):
    """Transcribe audio to text"""
    try:
        if audio_file is None:
            return "Please upload an audio file"
        
        files = {"file": open(audio_file, "rb")}
        data = {"model": model, "response_format": "json"}
        if language:
            data["language"] = language
        
        response = requests.post(
            f"{SERVER_URL}/v1/audio/transcriptions",
            files=files,
            data=data
        )
        
        files["file"].close()
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        result = response.json()
        return result.get("text", "")
        
    except Exception as e:
        return f"Error: {str(e)}"


def text_to_speech(text: str, model: str, voice: str, speed: float):
    """Convert text to speech"""
    try:
        if not text:
            return None, "Please enter text to convert"
        
        payload = {
            "model": model,
            "input": text,
            "voice": voice if voice else "alloy",
            "speed": speed,
            "response_format": "wav"
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/audio/speech",
            json=payload
        )
        
        if response.status_code != 200:
            return None, f"Error: {response.status_code} - {response.text}"
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(response.content)
            audio_path = tmp_file.name
        
        return audio_path, "Audio generated successfully"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# Image Generation
# ============================================================================

def generate_image(prompt: str, model: str, n: int, size: str, quality: str):
    """Generate images from text prompt"""
    try:
        if not prompt:
            return None, "Please enter a prompt"
        
        payload = {
            "model": model if model else None,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality if quality else "standard",
            "response_format": "b64_json"
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/images/generations",
            json=payload
        )
        
        if response.status_code != 200:
            return None, f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        images = []
        for img_data in data.get("data", []):
            if "b64_json" in img_data:
                img_bytes = base64.b64decode(img_data["b64_json"])
                img = PIL.Image.open(io.BytesIO(img_bytes))
                images.append(img)
        
        return images if images else None, "Images generated successfully" if images else "No images generated"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def edit_image(image, prompt: str, mask, model: str, n: int, size: str):
    """Edit an image"""
    try:
        if image is None:
            return None, "Please upload an image"
        if not prompt:
            return None, "Please enter an edit prompt"
        
        files = {}
        data = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json"
        }
        if model:
            data["model"] = model
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image.save(tmp_img, format="PNG")
            tmp_img_path = tmp_img.name
        
        files["image"] = open(tmp_img_path, "rb")
        
        if mask:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_mask:
                mask.save(tmp_mask, format="PNG")
                tmp_mask_path = tmp_mask.name
            files["mask"] = open(tmp_mask_path, "rb")
        
        response = requests.post(
            f"{SERVER_URL}/v1/images/edits",
            files=files,
            data=data
        )
        
        files["image"].close()
        if mask:
            files["mask"].close()
        os.unlink(tmp_img_path)
        if mask:
            os.unlink(tmp_mask_path)
        
        if response.status_code != 200:
            return None, f"Error: {response.status_code} - {response.text}"
        
        result = response.json()
        images = []
        for img_data in result.get("data", []):
            if "b64_json" in img_data:
                img_bytes = base64.b64decode(img_data["b64_json"])
                img = PIL.Image.open(io.BytesIO(img_bytes))
                images.append(img)
        
        return images if images else None, "Images edited successfully" if images else "No images generated"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def create_image_variations(image, model: str, n: int, size: str):
    """Create variations of an image"""
    try:
        if image is None:
            return None, "Please upload an image"
        
        files = {}
        data = {
            "n": n,
            "size": size,
            "response_format": "b64_json"
        }
        if model:
            data["model"] = model
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image.save(tmp_img, format="PNG")
            tmp_img_path = tmp_img.name
        
        files["image"] = open(tmp_img_path, "rb")
        
        response = requests.post(
            f"{SERVER_URL}/v1/images/variations",
            files=files,
            data=data
        )
        
        files["image"].close()
        os.unlink(tmp_img_path)
        
        if response.status_code != 200:
            return None, f"Error: {response.status_code} - {response.text}"
        
        result = response.json()
        images = []
        for img_data in result.get("data", []):
            if "b64_json" in img_data:
                img_bytes = base64.b64decode(img_data["b64_json"])
                img = PIL.Image.open(io.BytesIO(img_bytes))
                images.append(img)
        
        return images if images else None, "Image variations created successfully" if images else "No images generated"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# Embeddings
# ============================================================================

def create_embeddings(text: str, model: str, encoding_format: str):
    """Create text embeddings"""
    try:
        if not text:
            return "Please enter text"
        
        # Support multiple inputs (one per line)
        inputs = [line.strip() for line in text.strip().split("\n") if line.strip()]
        
        payload = {
            "model": model,
            "input": inputs if len(inputs) > 1 else inputs[0],
            "encoding_format": encoding_format if encoding_format else "float"
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/embeddings",
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        result = "Embeddings generated:\n\n"
        for idx, embedding_obj in enumerate(data.get("data", [])):
            embedding = embedding_obj.get("embedding", [])
            if encoding_format == "base64":
                result += f"Input {idx + 1}: {embedding[:100]}...\n"
            else:
                result += f"Input {idx + 1}: Vector dimension {len(embedding)}, first 10 values: {embedding[:10]}\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Moderation
# ============================================================================

def moderate_text(text: str, model: str):
    """Moderate text content"""
    try:
        if not text:
            return "Please enter text to moderate"
        
        payload = {
            "model": model if model else None,
            "input": text
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/moderations",
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        result = ""
        for idx, result_obj in enumerate(data.get("results", [])):
            flagged = result_obj.get("flagged", False)
            categories = result_obj.get("categories", {})
            scores = result_obj.get("category_scores", {})
            
            result += f"Input {idx + 1}:\n"
            result += f"  Flagged: {'⚠️ YES' if flagged else '✅ NO'}\n"
            result += f"  Categories:\n"
            for cat, value in categories.items():
                score = scores.get(cat.replace("_", "_"), 0)
                status = "⚠️" if value else "✅"
                result += f"    {status} {cat}: {score:.4f}\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# File Management
# ============================================================================

def upload_file(file, purpose: str):
    """Upload a file"""
    try:
        if file is None:
            return "Please select a file"
        
        files = {"file": open(file, "rb")}
        data = {"purpose": purpose if purpose else "assistants"}
        
        response = requests.post(
            f"{SERVER_URL}/v1/files",
            files=files,
            data=data
        )
        
        files["file"].close()
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        return f"File uploaded successfully!\nID: {data.get('id')}\nName: {data.get('filename')}\nSize: {data.get('bytes')} bytes"
        
    except Exception as e:
        return f"Error: {str(e)}"


def list_files():
    """List all uploaded files"""
    try:
        response = requests.get(f"{SERVER_URL}/v1/files")
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        files = data.get("data", [])
        
        if not files:
            return "No files uploaded"
        
        result = "Uploaded Files:\n\n"
        for f in files:
            result += f"ID: {f.get('id')}\n"
            result += f"Name: {f.get('filename')}\n"
            result += f"Size: {f.get('bytes')} bytes\n"
            result += f"Created: {f.get('created_at')}\n"
            result += "---\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Vector Store (RAG)
# ============================================================================

def add_document_to_vector_store(text: str, embedding_model: str, metadata: str):
    """Add document to vector store"""
    try:
        if not text:
            return "Please enter document text"
        
        payload = {
            "text": text,
            "embedding_model": embedding_model,
            "metadata": json.loads(metadata) if metadata and metadata.strip() else {}
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/vector_store/documents",
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        return f"Document added successfully!\nDocument ID: {data.get('doc_id')}\nText length: {data.get('text_length')} chars\nEmbedding dimension: {data.get('embedding_dimension')}"
        
    except Exception as e:
        return f"Error: {str(e)}"


def search_vector_store(query: str, embedding_model: str, top_k: int, threshold: float):
    """Search vector store"""
    try:
        if not query:
            return "Please enter a search query"
        
        payload = {
            "query": query,
            "embedding_model": embedding_model,
            "top_k": top_k,
            "threshold": threshold
        }
        
        response = requests.post(
            f"{SERVER_URL}/v1/vector_store/search",
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return "No results found"
        
        result_text = f"Found {len(results)} results:\n\n"
        for idx, result in enumerate(results):
            result_text += f"Result {idx + 1} (score: {result.get('score', 0):.4f}):\n"
            result_text += f"Text: {result.get('text', '')[:200]}...\n"
            if result.get('metadata'):
                result_text += f"Metadata: {result.get('metadata')}\n"
            result_text += "---\n"
        
        return result_text
        
    except Exception as e:
        return f"Error: {str(e)}"


def list_vector_store_documents():
    """List all documents in vector store"""
    try:
        response = requests.get(f"{SERVER_URL}/v1/vector_store/documents")
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        documents = data.get("documents", [])
        
        if not documents:
            return "No documents in vector store"
        
        result = f"Documents in Vector Store ({len(documents)}):\n\n"
        for doc in documents:
            result += f"ID: {doc.get('doc_id')}\n"
            result += f"Text: {doc.get('text', '')[:100]}...\n"
            result += f"Created: {doc.get('created_at')}\n"
            result += "---\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create the main Gradio interface"""
    
    # Get model lists
    model_types = get_model_types()
    
    # Common model dropdowns
    llm_models = model_types.get("llm", [])
    vlm_models = model_types.get("vlm", [])
    tts_models = model_types.get("tts", [])
    whisper_models = model_types.get("whisper", [])
    embedding_models = model_types.get("embedding", [])
    image_models = model_types.get("image", [])
    moderation_models = model_types.get("moderation", [])
    
    with gr.Blocks(title="OpenVINO GenAI Server - All Features", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 OpenVINO GenAI Server Interface")
        gr.Markdown("Access all features of the OpenVINO GenAI server through this interface.")
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat & Completions"):
                with gr.Row():
                    with gr.Column():
                        chat_messages = gr.Textbox(
                            label="Messages",
                            placeholder="Enter messages here. Format: user: message or just message",
                            lines=10
                        )
                        system_prompt_chat = gr.Textbox(
                            label="System Prompt (optional)",
                            placeholder="System instructions...",
                            lines=3
                        )
                        chat_model = gr.Dropdown(
                            choices=llm_models,
                            label="Model",
                            value=llm_models[0] if llm_models else None
                        )
                        with gr.Row():
                            chat_temperature = gr.Slider(0, 2, value=0.7, label="Temperature")
                            chat_max_tokens = gr.Slider(1, 4096, value=512, step=1, label="Max Tokens")
                        # Note: Streaming is disabled in UI for simplicity (non-streaming works better with Gradio)
                        chat_stream = gr.Checkbox(label="Stream (not supported in UI)", value=False, interactive=False)
                        chat_btn = gr.Button("Send", variant="primary")
                    
                    chat_output = gr.Textbox(label="Response", lines=15)
                
                chat_btn.click(
                    chat_completion,
                    inputs=[chat_messages, chat_model, chat_temperature, chat_max_tokens, chat_stream, system_prompt_chat],
                    outputs=chat_output
                )
                
                gr.Markdown("---")
                gr.Markdown("### Text Completion")
                
                with gr.Row():
                    with gr.Column():
                        completion_prompt = gr.Textbox(label="Prompt", lines=5)
                        completion_model = gr.Dropdown(
                            choices=llm_models,
                            label="Model",
                            value=llm_models[0] if llm_models else None
                        )
                        with gr.Row():
                            completion_temperature = gr.Slider(0, 2, value=0.7, label="Temperature")
                            completion_max_tokens = gr.Slider(1, 4096, value=512, step=1, label="Max Tokens")
                        completion_stream = gr.Checkbox(label="Stream (not supported in UI)", value=False, interactive=False)
                        completion_btn = gr.Button("Complete", variant="primary")
                    
                    completion_output = gr.Textbox(label="Completion", lines=10)
                
                completion_btn.click(
                    text_completion,
                    inputs=[completion_prompt, completion_model, completion_temperature, completion_max_tokens, completion_stream],
                    outputs=completion_output
                )
            
            # Vision Tab
            with gr.Tab("👁️ Vision Chat"):
                with gr.Row():
                    with gr.Column():
                        vision_image = gr.Image(label="Upload Image", type="pil")
                        vision_message = gr.Textbox(
                            label="Message",
                            placeholder="What would you like to know about this image?",
                            lines=3
                        )
                        vision_model = gr.Dropdown(
                            choices=vlm_models,
                            label="Model",
                            value=vlm_models[0] if vlm_models else None
                        )
                        with gr.Row():
                            vision_temperature = gr.Slider(0, 2, value=0.7, label="Temperature")
                            vision_max_tokens = gr.Slider(1, 4096, value=512, step=1, label="Max Tokens")
                        vision_btn = gr.Button("Analyze Image", variant="primary")
                    
                    vision_output = gr.Textbox(label="Response", lines=15)
                
                vision_btn.click(
                    vision_chat,
                    inputs=[vision_message, vision_image, vision_model, vision_temperature, vision_max_tokens],
                    outputs=vision_output
                )
            
            # Audio Tab
            with gr.Tab("🎤 Audio"):
                gr.Markdown("### Speech-to-Text (Transcription)")
                with gr.Row():
                    with gr.Column():
                        transcribe_audio_input = gr.Audio(label="Upload Audio", type="filepath")
                        transcribe_model = gr.Dropdown(
                            choices=whisper_models,
                            label="Model",
                            value=whisper_models[0] if whisper_models else None
                        )
                        transcribe_language = gr.Textbox(
                            label="Language (optional)",
                            placeholder="auto, en, es, fr, etc."
                        )
                        transcribe_btn = gr.Button("Transcribe", variant="primary")
                    
                    transcribe_output = gr.Textbox(label="Transcription", lines=10)
                
                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[transcribe_audio_input, transcribe_model, transcribe_language],
                    outputs=transcribe_output
                )
                
                gr.Markdown("---")
                gr.Markdown("### Text-to-Speech")
                with gr.Row():
                    with gr.Column():
                        tts_text = gr.Textbox(label="Text", lines=5)
                        tts_model = gr.Dropdown(
                            choices=tts_models,
                            label="Model",
                            value=tts_models[0] if tts_models else None
                        )
                        tts_voice = gr.Dropdown(
                            choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                            label="Voice",
                            value="alloy"
                        )
                        tts_speed = gr.Slider(0.25, 4.0, value=1.0, label="Speed")
                        tts_btn = gr.Button("Generate Speech", variant="primary")
                    
                    with gr.Column():
                        tts_audio_output = gr.Audio(label="Generated Audio")
                        tts_status = gr.Textbox(label="Status")
                
                tts_btn.click(
                    text_to_speech,
                    inputs=[tts_text, tts_model, tts_voice, tts_speed],
                    outputs=[tts_audio_output, tts_status]
                )
            
            # Images Tab
            with gr.Tab("🎨 Image Generation"):
                gr.Markdown("### Generate Images")
                with gr.Row():
                    with gr.Column():
                        image_prompt = gr.Textbox(label="Prompt", lines=3)
                        image_model = gr.Dropdown(
                            choices=image_models,
                            label="Model",
                            value=image_models[0] if image_models else None,
                            allow_custom_value=True
                        )
                        image_n = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                        image_size = gr.Dropdown(
                            choices=["256x256", "512x512", "1024x1024"],
                            label="Size",
                            value="1024x1024"
                        )
                        image_quality = gr.Dropdown(
                            choices=["standard", "hd"],
                            label="Quality",
                            value="standard"
                        )
                        image_gen_btn = gr.Button("Generate", variant="primary")
                    
                    image_gen_output = gr.Gallery(label="Generated Images", show_label=True)
                    image_gen_status = gr.Textbox(label="Status")
                
                image_gen_btn.click(
                    generate_image,
                    inputs=[image_prompt, image_model, image_n, image_size, image_quality],
                    outputs=[image_gen_output, image_gen_status]
                )
                
                gr.Markdown("---")
                gr.Markdown("### Edit Image")
                with gr.Row():
                    with gr.Column():
                        edit_image_input = gr.Image(label="Upload Image", type="pil")
                        edit_mask_input = gr.Image(label="Mask (optional)", type="pil")
                        edit_prompt = gr.Textbox(label="Edit Prompt", lines=2)
                        edit_model = gr.Dropdown(
                            choices=image_models,
                            label="Model",
                            value=image_models[0] if image_models else None,
                            allow_custom_value=True
                        )
                        edit_n = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                        edit_size = gr.Dropdown(
                            choices=["256x256", "512x512", "1024x1024"],
                            label="Size",
                            value="1024x1024"
                        )
                        edit_btn = gr.Button("Edit Image", variant="primary")
                    
                    edit_output = gr.Gallery(label="Edited Images", show_label=True)
                    edit_status = gr.Textbox(label="Status")
                
                edit_btn.click(
                    edit_image,
                    inputs=[edit_image_input, edit_prompt, edit_mask_input, edit_model, edit_n, edit_size],
                    outputs=[edit_output, edit_status]
                )
                
                gr.Markdown("---")
                gr.Markdown("### Create Variations")
                with gr.Row():
                    with gr.Column():
                        var_image_input = gr.Image(label="Upload Image", type="pil")
                        var_model = gr.Dropdown(
                            choices=image_models,
                            label="Model",
                            value=image_models[0] if image_models else None,
                            allow_custom_value=True
                        )
                        var_n = gr.Slider(1, 4, value=1, step=1, label="Number of Variations")
                        var_size = gr.Dropdown(
                            choices=["256x256", "512x512", "1024x1024"],
                            label="Size",
                            value="1024x1024"
                        )
                        var_btn = gr.Button("Create Variations", variant="primary")
                    
                    var_output = gr.Gallery(label="Variations", show_label=True)
                    var_status = gr.Textbox(label="Status")
                
                var_btn.click(
                    create_image_variations,
                    inputs=[var_image_input, var_model, var_n, var_size],
                    outputs=[var_output, var_status]
                )
            
            # Embeddings Tab
            with gr.Tab("📊 Embeddings"):
                with gr.Row():
                    with gr.Column():
                        embedding_text = gr.Textbox(
                            label="Text (one per line for multiple inputs)",
                            lines=5
                        )
                        embedding_model = gr.Dropdown(
                            choices=embedding_models,
                            label="Model",
                            value=embedding_models[0] if embedding_models else None
                        )
                        embedding_format = gr.Dropdown(
                            choices=["float", "base64"],
                            label="Encoding Format",
                            value="float"
                        )
                        embedding_btn = gr.Button("Generate Embeddings", variant="primary")
                    
                    embedding_output = gr.Textbox(label="Embeddings", lines=15)
                
                embedding_btn.click(
                    create_embeddings,
                    inputs=[embedding_text, embedding_model, embedding_format],
                    outputs=embedding_output
                )
            
            # Moderation Tab
            with gr.Tab("🛡️ Moderation"):
                with gr.Row():
                    with gr.Column():
                        moderation_text = gr.Textbox(
                            label="Text to Moderate",
                            lines=10
                        )
                        moderation_model = gr.Dropdown(
                            choices=moderation_models,
                            label="Model",
                            value=moderation_models[0] if moderation_models else None,
                            allow_custom_value=True
                        )
                        moderation_btn = gr.Button("Moderate", variant="primary")
                    
                    moderation_output = gr.Textbox(label="Moderation Results", lines=15)
                
                moderation_btn.click(
                    moderate_text,
                    inputs=[moderation_text, moderation_model],
                    outputs=moderation_output
                )
            
            # Files Tab
            with gr.Tab("📁 Files"):
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(label="Upload File")
                        file_purpose = gr.Dropdown(
                            choices=["assistants", "fine-tune", "batch", "vision"],
                            label="Purpose",
                            value="assistants"
                        )
                        file_upload_btn = gr.Button("Upload", variant="primary")
                    
                    file_upload_output = gr.Textbox(label="Upload Result", lines=5)
                
                file_upload_btn.click(
                    upload_file,
                    inputs=[file_upload, file_purpose],
                    outputs=file_upload_output
                )
                
                gr.Markdown("---")
                gr.Markdown("### List Files")
                
                list_files_btn = gr.Button("List All Files", variant="primary")
                list_files_output = gr.Textbox(label="Files", lines=15)
                
                list_files_btn.click(
                    list_files,
                    outputs=list_files_output
                )
            
            # Vector Store Tab
            with gr.Tab("🗄️ Vector Store (RAG)"):
                gr.Markdown("### Add Document")
                with gr.Row():
                    with gr.Column():
                        vector_text = gr.Textbox(
                            label="Document Text",
                            lines=5
                        )
                        vector_embedding_model = gr.Dropdown(
                            choices=embedding_models,
                            label="Embedding Model",
                            value=embedding_models[0] if embedding_models else None
                        )
                        vector_metadata = gr.Textbox(
                            label="Metadata (JSON, optional)",
                            placeholder='{"source": "example", "category": "docs"}',
                            lines=3
                        )
                        vector_add_btn = gr.Button("Add Document", variant="primary")
                    
                    vector_add_output = gr.Textbox(label="Result", lines=5)
                
                vector_add_btn.click(
                    add_document_to_vector_store,
                    inputs=[vector_text, vector_embedding_model, vector_metadata],
                    outputs=vector_add_output
                )
                
                gr.Markdown("---")
                gr.Markdown("### Search")
                with gr.Row():
                    with gr.Column():
                        vector_query = gr.Textbox(
                            label="Search Query",
                            lines=2
                        )
                        vector_search_model = gr.Dropdown(
                            choices=embedding_models,
                            label="Embedding Model",
                            value=embedding_models[0] if embedding_models else None
                        )
                        vector_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
                        vector_threshold = gr.Slider(0, 1, value=0.5, label="Similarity Threshold")
                        vector_search_btn = gr.Button("Search", variant="primary")
                    
                    vector_search_output = gr.Textbox(label="Search Results", lines=15)
                
                vector_search_btn.click(
                    search_vector_store,
                    inputs=[vector_query, vector_search_model, vector_top_k, vector_threshold],
                    outputs=vector_search_output
                )
                
                gr.Markdown("---")
                gr.Markdown("### List Documents")
                
                vector_list_btn = gr.Button("List All Documents", variant="primary")
                vector_list_output = gr.Textbox(label="Documents", lines=15)
                
                vector_list_btn.click(
                    list_vector_store_documents,
                    outputs=vector_list_output
                )
        
        gr.Markdown("---")
        gr.Markdown("### Server Status")
        with gr.Row():
            status_btn = gr.Button("Check Server Status", variant="secondary")
            status_output = gr.Textbox(label="Status", lines=5)
        
        def check_status():
            try:
                response = requests.get(f"{SERVER_URL}/health")
                if response.status_code == 200:
                    data = response.json()
                    return f"✅ Server is healthy!\n\nModels loaded: {data.get('models_loaded', 0)}\nLLM: {data.get('llm_models', 0)}\nVLM: {data.get('vlm_models', 0)}\nWhisper: {data.get('whisper_models', 0)}\nTTS: {data.get('tts_models', 0)}\nEmbeddings: {data.get('embedding_models', 0)}\nImage: {data.get('text2image_models', 0)}\nModeration: {data.get('moderation_models', 0)}\nFiles: {data.get('files_stored', 0)}\nVector Store: {data.get('documents_in_vector_store', 0)}"
                else:
                    return f"⚠️ Server returned status {response.status_code}"
            except Exception as e:
                return f"❌ Cannot connect to server at {SERVER_URL}\nError: {str(e)}\n\nMake sure the server is running with: python npu.py"
        
        status_btn.click(check_status, outputs=status_output)
    
    return app


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

