"""
Utility Functions
Helper functions for prompt formatting, tool parsing, JSON handling, etc.
"""

import json
import uuid
import re
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import openvino as ov
import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .models import ChatMessage, ToolDefinition, ToolCall, FunctionCall


# ============================================================================
# Error Handling
# ============================================================================

def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    status_code: int = 400
) -> JSONResponse:
    """Create OpenAI-compatible error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code
            }
        }
    )


# ============================================================================
# Tool Calling Helpers
# ============================================================================

def format_tools_for_prompt(tools: List[ToolDefinition]) -> str:
    """Format tools/functions for inclusion in the prompt"""
    if not tools:
        return ""
    
    tools_description = "\n\n# Tools\n\nYou have access to the following tools:\n\n"
    for tool in tools:
        func = tool.function
        tools_description += f"- {func.name}: {func.description or 'No description'}\n"
        if func.parameters:
            tools_description += f"  Parameters: {json.dumps(func.parameters, indent=2)}\n"
    
    tools_description += "\nTo use a tool, respond with a JSON object in this format:\n"
    tools_description += '{"tool_calls": [{"name": "function_name", "arguments": {...}}]}\n'
    
    return tools_description


def parse_tool_calls_from_response(response_text: str) -> Tuple[str, List[ToolCall]]:
    """Parse tool calls from model response"""
    tool_calls = []
    cleaned_text = response_text
    
    try:
        # Look for tool_call XML tags (Qwen format)
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=FunctionCall(
                        name=tool_data.get("name", ""),
                        arguments=json.dumps(tool_data.get("arguments", {}))
                    )
                )
                tool_calls.append(tool_call)
                cleaned_text = cleaned_text.replace(f"<tool_call>{match}</tool_call>", "")
            except json.JSONDecodeError:
                pass
        
        # Also try to parse direct JSON format
        if not tool_calls and "{" in response_text and "tool_calls" in response_text:
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    data = json.loads(json_str)
                    
                    if "tool_calls" in data:
                        for tc in data["tool_calls"]:
                            tool_call = ToolCall(
                                id=f"call_{uuid.uuid4().hex[:24]}",
                                type="function",
                                function=FunctionCall(
                                    name=tc.get("name", ""),
                                    arguments=json.dumps(tc.get("arguments", {}))
                                )
                            )
                            tool_calls.append(tool_call)
                        cleaned_text = response_text[:start] + response_text[end:]
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"Warning: Error parsing tool calls: {e}")
    
    return cleaned_text.strip(), tool_calls


# ============================================================================
# JSON Handling
# ============================================================================

def validate_json_schema(response_text: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate response against JSON schema"""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return False, "No JSON object found in response"
        
        json_str = json_match.group(0)
        data = json.loads(json_str)
        
        if "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if schema.get("required", []) and prop in schema["required"]:
                    if prop not in data:
                        return False, f"Required property '{prop}' missing"
        
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def extract_json_from_response(response_text: str) -> Optional[str]:
    """Extract JSON object from response text"""
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            json.loads(json_str)  # Validate
            return json_str
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
# Prompt Building
# ============================================================================

def build_chat_prompt(messages: List[ChatMessage], tools: Optional[List[ToolDefinition]] = None) -> str:
    """Build a properly formatted chat prompt from messages"""
    prompt_parts = []
    
    # Add tools description if provided
    if tools:
        tools_info = format_tools_for_prompt(tools)
        prompt_parts.append(f"<|im_start|>system\n{tools_info}<|im_end|>\n")
    
    for msg in messages:
        role = msg.role
        
        if role == "system":
            content = msg.content if isinstance(msg.content, str) else ""
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        
        elif role == "user":
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                text_parts = [part.text for part in msg.content if part.type == "text" and part.text]
                content = " ".join(text_parts)
            else:
                content = ""
            
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        
        elif role == "assistant":
            content = msg.content if isinstance(msg.content, str) else ""
            
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content += f"\n<tool_call>\n{{'name': '{tc.function.name}', 'arguments': {tc.function.arguments}}}\n</tool_call>"
            
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        
        elif role == "tool":
            content = msg.content if isinstance(msg.content, str) else ""
            prompt_parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
    
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "".join(prompt_parts)


def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string (text-only)"""
    formatted = ""
    for msg in messages:
        content_text = ""
        
        if isinstance(msg.content, str):
            content_text = msg.content
        elif isinstance(msg.content, list):
            text_parts = [part.text for part in msg.content if part.type == "text" and part.text]
            content_text = " ".join(text_parts)
        
        if msg.role == "system":
            formatted += f"System: {content_text}\n\n"
        elif msg.role == "user":
            formatted += f"User: {content_text}\n\n"
        elif msg.role == "assistant":
            formatted += f"Assistant: {content_text}\n\n"
    
    formatted += "Assistant: "
    return formatted


# ============================================================================
# Image Loading
# ============================================================================

def load_image_from_url(url: str, file_storage=None) -> ov.Tensor:
    """Load image from URL or base64 data URI"""
    import requests
    
    if url.startswith('data:image'):
        # Base64 encoded image - 100% LOCAL
        header, encoded = url.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
    elif url.startswith('http://') or url.startswith('https://'):
        # Remote URL - ONLY EXTERNAL NETWORK CALL!
        # Prefer using base64 or file:// for 100% local operation
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    elif url.startswith('file://'):
        # Local file - 100% LOCAL
        file_path = url[7:]
        image = Image.open(file_path)
    elif url.startswith('file-'):
        # Uploaded file ID - 100% LOCAL
        if not file_storage:
            raise HTTPException(status_code=500, detail="File storage not initialized")
        file_path = file_storage.get_file_path(url)
        if not file_path:
            raise HTTPException(status_code=404, detail=f"File '{url}' not found")
        image = Image.open(file_path)
    else:
        # Assume local file path - 100% LOCAL
        image = Image.open(url)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image)
    return ov.Tensor(image_array)


def extract_content_parts(messages: List[ChatMessage], file_storage=None) -> Tuple[str, List[ov.Tensor], List[str]]:
    """Extract text, images, and file references from messages"""
    text_parts = []
    images = []
    file_ids = []
    
    for msg in messages:
        role_prefix = ""
        if msg.role == "system":
            role_prefix = "System: "
        elif msg.role == "user":
            role_prefix = "User: "
        elif msg.role == "assistant":
            role_prefix = "Assistant: "
        
        if isinstance(msg.content, str):
            text_parts.append(f"{role_prefix}{msg.content}\n\n")
        
        elif isinstance(msg.content, list):
            msg_text = []
            for part in msg.content:
                if part.type == "text" and part.text:
                    msg_text.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    try:
                        img_tensor = load_image_from_url(part.image_url.url, file_storage)
                        images.append(img_tensor)
                        msg_text.append("[Image attached]")
                        
                        if part.image_url.url.startswith('file-'):
                            file_ids.append(part.image_url.url)
                    except Exception as e:
                        msg_text.append(f"[Image load error: {e}]")
            
            if msg_text:
                text_parts.append(f"{role_prefix}{' '.join(msg_text)}\n\n")
    
    text_parts.append("Assistant: ")
    full_prompt = "".join(text_parts)
    
    return full_prompt, images, file_ids

