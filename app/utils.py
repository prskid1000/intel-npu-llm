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

def detect_model_family(model_name: str) -> str:
    """
    Detect the model family from model name for tool calling format
    
    Returns:
        Model family: 'qwen', 'llama', 'mistral', 'phi', or 'generic'
    """
    model_lower = model_name.lower()
    
    if 'qwen' in model_lower:
        return 'qwen'
    elif 'llama' in model_lower or 'llama-3' in model_lower:
        return 'llama'
    elif 'mistral' in model_lower or 'mixtral' in model_lower:
        return 'mistral'
    elif 'phi' in model_lower:
        return 'phi'
    else:
        return 'generic'


def format_tools_for_qwen(tools: List[ToolDefinition]) -> str:
    """Format tools for Qwen models (XML-style tags)"""
    tools_list = []
    for tool in tools:
        func = tool.function
        tool_dict = {
            "type": "function",
            "function": {
                "name": func.name,
                "description": func.description or "No description provided",
                "parameters": func.parameters or {}
            }
        }
        tools_list.append(tool_dict)
    
    tools_description = "\n\n# Available Tools\n\n"
    tools_description += "You have access to the following functions. Use them when necessary:\n\n"
    tools_description += f"{json.dumps(tools_list, indent=2)}\n\n"
    tools_description += "To call a function, respond with:\n"
    tools_description += '<tool_call>\n{"name": "function_name", "arguments": {"param": "value"}}\n</tool_call>\n\n'
    tools_description += "You can call multiple functions by using multiple <tool_call> tags.\n"
    tools_description += "After receiving function results, use them to answer the user's question.\n"
    
    return tools_description


def format_tools_for_llama(tools: List[ToolDefinition]) -> str:
    """Format tools for Llama 3.1+ models (JSON array format)"""
    tools_list = []
    for tool in tools:
        func = tool.function
        tool_dict = {
            "type": "function",
            "function": {
                "name": func.name,
                "description": func.description or "No description provided",
                "parameters": func.parameters or {}
            }
        }
        tools_list.append(tool_dict)
    
    tools_description = "\n\nYou have access to the following functions:\n\n"
    tools_description += f"{json.dumps(tools_list, indent=2)}\n\n"
    tools_description += "To call a function, respond ONLY with a JSON object in this exact format:\n"
    tools_description += '{"name": "function_name", "parameters": {"param": "value"}}\n\n'
    tools_description += "For multiple function calls, use a JSON array:\n"
    tools_description += '[{"name": "func1", "parameters": {...}}, {"name": "func2", "parameters": {...}}]\n\n'
    tools_description += "After receiving function results, respond with the answer to the user's question.\n"
    
    return tools_description


def format_tools_for_mistral(tools: List[ToolDefinition]) -> str:
    """Format tools for Mistral/Mixtral models"""
    tools_list = []
    for tool in tools:
        func = tool.function
        tool_dict = {
            "type": "function",
            "function": {
                "name": func.name,
                "description": func.description or "No description provided",
                "parameters": func.parameters or {}
            }
        }
        tools_list.append(tool_dict)
    
    tools_description = "\n\n[AVAILABLE_TOOLS]\n"
    tools_description += f"{json.dumps(tools_list, indent=2)}\n"
    tools_description += "[/AVAILABLE_TOOLS]\n\n"
    tools_description += "To use a tool, respond with:\n"
    tools_description += '[TOOL_CALLS] [{"name": "function_name", "arguments": {...}}]\n\n'
    tools_description += "You can call multiple tools in the array.\n"
    tools_description += "After receiving tool results, provide your final answer.\n"
    
    return tools_description


def format_tools_for_phi(tools: List[ToolDefinition]) -> str:
    """Format tools for Phi models (Python-style function signatures)"""
    tools_list = []
    for tool in tools:
        func = tool.function
        tool_dict = {
            "type": "function",
            "function": {
                "name": func.name,
                "description": func.description or "No description provided",
                "parameters": func.parameters or {}
            }
        }
        tools_list.append(tool_dict)
    
    tools_description = "\n\n## Available Functions\n\n"
    tools_description += "You have access to the following functions:\n\n"
    tools_description += f"{json.dumps(tools_list, indent=2)}\n\n"
    tools_description += "To call a function, use this format:\n"
    tools_description += '```function\n{"name": "function_name", "arguments": {"param": "value"}}\n```\n\n'
    tools_description += "You can call multiple functions. After receiving results, answer the user's question.\n"
    
    return tools_description


def format_tools_for_generic(tools: List[ToolDefinition]) -> str:
    """Format tools for generic models (simple instruction format)"""
    tools_list = []
    for tool in tools:
        func = tool.function
        tools_list.append({
            "name": func.name,
            "description": func.description or "No description provided",
            "parameters": func.parameters or {}
        })
    
    tools_description = "\n\nYou have access to these functions:\n\n"
    for tool_info in tools_list:
        tools_description += f"Function: {tool_info['name']}\n"
        tools_description += f"Description: {tool_info['description']}\n"
        tools_description += f"Parameters: {json.dumps(tool_info['parameters'], indent=2)}\n\n"
    
    tools_description += "To call a function, respond with JSON:\n"
    tools_description += '{"function": "function_name", "arguments": {"param": "value"}}\n\n'
    
    return tools_description


def format_tools_for_prompt(tools: List[ToolDefinition], model_name: str = None) -> str:
    """
    Format tools/functions for inclusion in the prompt (model-aware)
    
    Args:
        tools: List of tool definitions
        model_name: Name of the model to format for (auto-detects if provided)
    
    Returns:
        Formatted tools string for the specific model
    """
    if not tools:
        return ""
    
    # Detect model family
    model_family = detect_model_family(model_name) if model_name else 'generic'
    
    print(f"📝 Formatting {len(tools)} tool(s) for model family: {model_family}")
    
    # Use appropriate formatter
    if model_family == 'qwen':
        formatted = format_tools_for_qwen(tools)
    elif model_family == 'llama':
        formatted = format_tools_for_llama(tools)
    elif model_family == 'mistral':
        formatted = format_tools_for_mistral(tools)
    elif model_family == 'phi':
        formatted = format_tools_for_phi(tools)
    else:
        formatted = format_tools_for_generic(tools)
    
    print(f"📝 Formatted tools prompt length: {len(formatted)} characters")
    return formatted


def parse_tool_calls_from_response(response_text: str, model_name: str = None) -> Tuple[str, List[ToolCall]]:
    """
    Parse tool calls from model response (supports multiple formats)
    
    Tries to detect and parse tool calls in various formats:
    - Qwen: <tool_call>...</tool_call>
    - Llama: {"name": "...", "parameters": {...}}
    - Mistral: [TOOL_CALLS] [...]
    - Phi: ```function...```
    - Generic: {"function": "...", "arguments": {...}}
    """
    tool_calls = []
    cleaned_text = response_text
    model_family = detect_model_family(model_name) if model_name else 'generic'
    
    try:
        # 1. Try Qwen format: <tool_call>...</tool_call>
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
        
        if matches:
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
        
        # 2. Try Mistral format: [TOOL_CALLS] [...]
        if not tool_calls:
            mistral_pattern = r'\[TOOL_CALLS\]\s*(\[.*?\])'
            mistral_matches = re.findall(mistral_pattern, response_text, re.DOTALL)
            
            if mistral_matches:
                try:
                    tools_array = json.loads(mistral_matches[0])
                    for tool_data in tools_array:
                        tool_call = ToolCall(
                            id=f"call_{uuid.uuid4().hex[:24]}",
                            type="function",
                            function=FunctionCall(
                                name=tool_data.get("name", ""),
                                arguments=json.dumps(tool_data.get("arguments", {}))
                            )
                        )
                        tool_calls.append(tool_call)
                    cleaned_text = re.sub(mistral_pattern, "", response_text, flags=re.DOTALL)
                except json.JSONDecodeError:
                    pass
        
        # 3. Try Phi format: ```function...```
        if not tool_calls:
            phi_pattern = r'```function\s*(.*?)\s*```'
            phi_matches = re.findall(phi_pattern, response_text, re.DOTALL)
            
            if phi_matches:
                for match in phi_matches:
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
                        cleaned_text = cleaned_text.replace(f"```function\n{match}\n```", "")
                    except json.JSONDecodeError:
                        pass
        
        # 4. Try Llama format: JSON object(s) with "name" and "parameters"
        if not tool_calls:
            # Try array format first
            json_array_pattern = r'\[(\s*\{[^]]+\}(?:\s*,\s*\{[^]]+\})*)\s*\]'
            array_matches = re.findall(json_array_pattern, response_text, re.DOTALL)
            
            if array_matches:
                try:
                    tools_array = json.loads(f"[{array_matches[0]}]")
                    for tool_data in tools_array:
                        if "name" in tool_data:
                            tool_call = ToolCall(
                                id=f"call_{uuid.uuid4().hex[:24]}",
                                type="function",
                                function=FunctionCall(
                                    name=tool_data.get("name", ""),
                                    arguments=json.dumps(tool_data.get("parameters", tool_data.get("arguments", {})))
                                )
                            )
                            tool_calls.append(tool_call)
                    if tool_calls:
                        cleaned_text = re.sub(json_array_pattern, "", response_text, flags=re.DOTALL)
                except json.JSONDecodeError:
                    pass
            
            # Try single object format
            if not tool_calls:
                json_pattern = r'\{[^}]*"name"\s*:\s*"[^"]+"[^}]*\}'
                json_matches = re.findall(json_pattern, response_text, re.DOTALL)
                
                for match in json_matches:
                    try:
                        tool_data = json.loads(match)
                        if "name" in tool_data:
                            tool_call = ToolCall(
                                id=f"call_{uuid.uuid4().hex[:24]}",
                                type="function",
                                function=FunctionCall(
                                    name=tool_data.get("name", ""),
                                    arguments=json.dumps(tool_data.get("parameters", tool_data.get("arguments", {})))
                                )
                            )
                            tool_calls.append(tool_call)
                            cleaned_text = cleaned_text.replace(match, "")
                    except json.JSONDecodeError:
                        pass
        
        # 5. Try generic format: {"function": "...", "arguments": {...}}
        if not tool_calls:
            generic_pattern = r'\{\s*"function"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
            generic_matches = re.findall(generic_pattern, response_text, re.DOTALL)
            
            for match in generic_matches:
                try:
                    tool_data = json.loads(match)
                    tool_call = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        type="function",
                        function=FunctionCall(
                            name=tool_data.get("function", ""),
                            arguments=json.dumps(tool_data.get("arguments", {}))
                        )
                    )
                    tool_calls.append(tool_call)
                    cleaned_text = cleaned_text.replace(match, "")
                except json.JSONDecodeError:
                    pass
        
        # 6. Legacy format: {"tool_calls": [...]}
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
# Stop Sequence Handling
# ============================================================================

def apply_stop_sequences(text: str, stop_sequences: List[str]) -> Tuple[str, bool]:
    """
    Apply stop sequences to generated text (post-processing workaround)
    
    Performs case-insensitive matching for better compatibility.
    
    Returns:
        Tuple of (processed_text, was_stopped)
    """
    if not stop_sequences or not text:
        return text, False
    
    # Find the earliest occurrence of any stop sequence (case-insensitive)
    earliest_pos = len(text)
    found_stop = None
    text_lower = text.lower()
    
    for stop_seq in stop_sequences:
        stop_seq_lower = stop_seq.lower()
        pos = text_lower.find(stop_seq_lower)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            found_stop = stop_seq
    
    if found_stop:
        # Truncate at stop sequence (preserve original case)
        truncated = text[:earliest_pos].rstrip()
        return truncated, True
    
    return text, False


# ============================================================================
# Prompt Building
# ============================================================================

def build_chat_prompt(messages: List[ChatMessage], tools: Optional[List[ToolDefinition]] = None, model_name: str = None) -> str:
    """Build a properly formatted chat prompt from messages (model-aware)"""
    prompt_parts = []
    
    # Collect system messages and tools
    system_content = ""
    if tools:
        system_content = format_tools_for_prompt(tools, model_name)
    
    for msg in messages:
        role = msg.role
        
        if role == "system":
            content = msg.content if isinstance(msg.content, str) else ""
            if system_content:
                # Merge system message with tools
                system_content = content + "\n" + system_content
            else:
                system_content = content
        
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
            
            # Format tool calls in Qwen format
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content += f"\n<tool_call>\n{json.dumps({'name': tc.function.name, 'arguments': json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments})}\n</tool_call>"
            
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        
        elif role == "tool":
            content = msg.content if isinstance(msg.content, str) else ""
            tool_call_id = getattr(msg, 'tool_call_id', 'unknown')
            prompt_parts.append(f"<|im_start|>user\n<tool_response id=\"{tool_call_id}\">\n{content}\n</tool_response><|im_end|>\n")
    
    # Add system content at the beginning if present
    if system_content:
        prompt_parts.insert(0, f"<|im_start|>system\n{system_content}<|im_end|>\n")
        # Log a preview of the tools section
        if "Available Tools" in system_content or "Available Functions" in system_content:
            preview_lines = system_content.split('\n')[:10]  # First 10 lines
            print(f"📝 Tools section preview (first 10 lines):")
            for line in preview_lines:
                print(f"   {line}")
    
    prompt_parts.append("<|im_start|>assistant\n")
    
    full_prompt = "".join(prompt_parts)
    if tools:
        print(f"📝 Full prompt length: {len(full_prompt)} characters")
        # Show where tools appear in prompt
        if "Available Tools" in full_prompt or "Available Functions" in full_prompt:
            print(f"✅ Tools section confirmed in prompt")
    
    return full_prompt


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
    """Extract text, images, and file references from messages - VLM format with Phi-3 chat template"""
    text_parts = []
    images = []
    file_ids = []
    
    for msg in messages:
        # Use Phi-3 chat template format: <|role|>\ncontent<|end|>\n
        role = msg.role
        
        if isinstance(msg.content, str):
            text_parts.append(f"<|{role}|>\n{msg.content}<|end|>\n")
        
        elif isinstance(msg.content, list):
            msg_text = []
            for part in msg.content:
                if part.type == "text" and part.text:
                    msg_text.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    try:
                        img_tensor = load_image_from_url(part.image_url.url, file_storage)
                        images.append(img_tensor)
                        msg_text.append("<image>")  # Phi-3 Vision image placeholder
                        
                        if part.image_url.url.startswith('file-'):
                            file_ids.append(part.image_url.url)
                    except Exception as e:
                        msg_text.append(f"[Image load error: {e}]")
            
            if msg_text:
                content = ' '.join(msg_text)
                text_parts.append(f"<|{role}|>\n{content}<|end|>\n")
    
    # Add assistant prompt to continue generation
    text_parts.append("<|assistant|>\n")
    full_prompt = "".join(text_parts)
    
    return full_prompt, images, file_ids

