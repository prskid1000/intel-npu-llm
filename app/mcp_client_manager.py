"""
MCP Client Manager
Manages connections to multiple MCP (Model Context Protocol) servers
Supports STDIO and HTTP transports
"""

import asyncio
import subprocess
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from urllib.parse import urlparse
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """Server connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"


@dataclass
class ServerState:
    """State for an MCP server connection"""
    server_id: str
    config: Dict[str, Any]
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    client: Optional[Any] = None  # MCP client instance
    transport: Optional[Any] = None  # Transport instance
    process: Optional[subprocess.Popen] = None  # For STDIO transport
    session: Optional[aiohttp.ClientSession] = None  # For HTTP transport
    tools_cache: List[Dict[str, Any]] = field(default_factory=list)
    resources_cache: List[Dict[str, Any]] = field(default_factory=list)
    prompts_cache: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    _read_lock: Optional[asyncio.Lock] = None  # Lock for serializing stdout reads


class MCPClientManager:
    """Manages connections to multiple MCP servers"""
    
    def __init__(self, servers: Optional[Dict[str, Dict[str, Any]]] = None, options: Optional[Dict[str, Any]] = None):
        """
        Initialize MCP client manager
        
        Args:
            servers: Dictionary of server_id -> server_config
            options: Manager options (timeout, etc.)
        """
        self.server_states: Dict[str, ServerState] = {}
        self.default_timeout = (options or {}).get("default_timeout", 30000)  # 30 seconds
        self.default_client_name = (options or {}).get("default_client_name", "npu-mcp-client")
        self.default_client_version = (options or {}).get("default_client_version", "1.0.0")
        
        # Initialize servers if provided
        if servers:
            for server_id, config in servers.items():
                self.server_states[server_id] = ServerState(
                    server_id=server_id,
                    config=config,
                    status=ConnectionStatus.DISCONNECTED
                )
    
    def list_servers(self) -> List[str]:
        """List all server IDs"""
        return list(self.server_states.keys())
    
    def has_server(self, server_id: str) -> bool:
        """Check if server exists"""
        return server_id in self.server_states
    
    def get_server_summaries(self) -> List[Dict[str, Any]]:
        """Get summary of all servers"""
        summaries = []
        for server_id, state in self.server_states.items():
            summaries.append({
                "id": server_id,
                "status": state.status.value,
                "config": state.config,
                "error": state.error
            })
        return summaries
    
    def get_connection_status(self, server_id: str) -> str:
        """Get connection status for a server"""
        state = self.server_states.get(server_id)
        if not state:
            return ConnectionStatus.DISCONNECTED.value
        return state.status.value
    
    def is_stdio_config(self, config: Dict[str, Any]) -> bool:
        """Check if config is for STDIO transport"""
        return "command" in config
    
    async def connect_to_server(self, server_id: str, config: Dict[str, Any]) -> None:
        """
        Connect to an MCP server
        
        Args:
            server_id: Unique identifier for the server
            config: Server configuration (command/args for STDIO or url for HTTP)
        """
        # Check if already connected
        if server_id in self.server_states:
            state = self.server_states[server_id]
            if state.status == ConnectionStatus.CONNECTED:
                raise ValueError(f"MCP server '{server_id}' is already connected")
        
        # Create or update state
        if server_id not in self.server_states:
            self.server_states[server_id] = ServerState(
                server_id=server_id,
                config=config,
                status=ConnectionStatus.CONNECTING,
                _read_lock=asyncio.Lock()
            )
        else:
            state = self.server_states[server_id]
            state.config = config
            state.status = ConnectionStatus.CONNECTING
            state.error = None
            if state._read_lock is None:
                state._read_lock = asyncio.Lock()
        
        state = self.server_states[server_id]
        
        try:
            if self.is_stdio_config(config):
                await self._connect_via_stdio(server_id, state, config)
            else:
                await self._connect_via_http(server_id, state, config)
            
            # Verify connection is actually working by checking if we can initialize
            # If initialization fails, don't mark as connected
            try:
                # Initialize is already called in _connect_via_stdio, but verify it worked
                if state.transport == "stdio" and state.process:
                    # Verify process is still running
                    if state.process.returncode is not None:
                        raise Exception(f"Process exited with code {state.process.returncode}")
                
                state.status = ConnectionStatus.CONNECTED
                logger.info(f"✅ Connected to MCP server: {server_id}")
                
                # Fetch tools immediately after connection
                # If this fails, we still consider it connected but log the warning
                try:
                    await self._fetch_server_tools(server_id)
                except Exception as e:
                    logger.warning(f"Failed to fetch tools from server '{server_id}' after connection: {e}")
                    # Don't fail connection if tools can't be fetched - server might still be usable
            
            except Exception as init_error:
                # If verification fails, disconnect and raise
                await self._cleanup_connection(server_id, state)
                raise Exception(f"Connection verification failed: {str(init_error)}")
            
        except Exception as e:
            state.status = ConnectionStatus.DISCONNECTED
            state.error = str(e)
            logger.error(f"❌ Failed to connect to MCP server '{server_id}': {e}")
            # Cleanup any partial connection
            await self._cleanup_connection(server_id, state)
            raise
    
    async def _cleanup_connection(self, server_id: str, state: ServerState) -> None:
        """Cleanup a failed connection"""
        try:
            if state.process:
                try:
                    state.process.terminate()
                    await asyncio.wait_for(state.process.wait(), timeout=2.0)
                except:
                    try:
                        state.process.kill()
                        await state.process.wait()
                    except:
                        pass
                state.process = None
            
            if state.session:
                try:
                    await state.session.close()
                except:
                    pass
                state.session = None
            
            state.transport = None
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
    
    async def _connect_via_stdio(self, server_id: str, state: ServerState, config: Dict[str, Any]) -> None:
        """Connect via STDIO transport"""
        import os
        import sys
        import shutil
        import asyncio
        
        command = config["command"]
        args = config.get("args", [])
        env = config.get("env", {})
        
        # Merge with default environment
        full_env = {**os.environ, **env}
        
        # On Windows, handle commands that might need shell resolution
        # Check if command exists in PATH
        command_path = shutil.which(command)
        if not command_path:
            # On Windows, try with .cmd/.bat extensions
            if sys.platform == "win32":
                for ext in [".cmd", ".bat", ".exe"]:
                    test_path = shutil.which(f"{command}{ext}")
                    if test_path:
                        command_path = test_path
                        break
                
                # If still not found, try using shell=True for Windows
                if not command_path:
                    # Use shell=True on Windows to let cmd.exe resolve the command
                    use_shell = True
                else:
                    use_shell = False
            else:
                raise Exception(f"Command '{command}' not found in PATH. Make sure it's installed and available.")
        else:
            use_shell = False
        
        try:
            # Build command list
            if use_shell and sys.platform == "win32":
                # On Windows with shell=True, combine command and args into a single string
                cmd_str = command
                if args:
                    cmd_str += " " + " ".join(str(arg) for arg in args)
                cmd = cmd_str
            else:
                # Use list format (more secure)
                cmd = [command_path or command] + (args if isinstance(args, list) else [])
            
            # Use asyncio subprocess for better async handling
            # Build command list properly
            cmd_list = [command_path or command]
            if args and isinstance(args, list):
                cmd_list.extend(args)
            elif args and not isinstance(args, list):
                # If args is a string, split it
                cmd_list.extend(str(args).split())
            
            # For Windows with shell commands, we might need different handling
            if use_shell and sys.platform == "win32":
                # On Windows, use shell=True with create_subprocess_shell
                cmd_str = command
                if args:
                    if isinstance(args, list):
                        cmd_str += " " + " ".join(str(a) for a in args)
                    else:
                        cmd_str += " " + str(args)
                
                process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=full_env
                )
            else:
                # Use create_subprocess_exec for better security
                process = await asyncio.create_subprocess_exec(
                    *cmd_list,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=full_env
                )
            
            state.process = process
            state.transport = "stdio"
            
            # Send initialize request
            await self._send_initialize(server_id, state)
            
            logger.info(f"Started STDIO process for server: {server_id} (command: {command})")
            
        except FileNotFoundError as e:
            raise Exception(f"Command '{command}' not found. Make sure it's installed and in your PATH. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to start STDIO process: {str(e)}")
    
    async def _read_jsonrpc_response(
        self,
        stdout_stream,
        timeout: float = 5.0,
        operation: str = "operation",
        max_attempts: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Read JSON-RPC response from stdout, skipping non-JSON lines
        
        Args:
            stdout_stream: The stdout stream to read from
            timeout: Timeout per line read
            operation: Operation name for error messages
            max_attempts: Maximum number of lines to read
            
        Returns:
            JSON-RPC response dict or None if not found
        """
        response = None
        
        for attempt in range(max_attempts):
            try:
                line_bytes = await asyncio.wait_for(
                    stdout_stream.readline(),
                    timeout=timeout
                )
                
                if not line_bytes:
                    if attempt == 0:
                        logger.warning(f"No response from MCP server for {operation}")
                        return None
                    break
                
                # Decode bytes to string
                line = line_bytes.decode('utf-8') if isinstance(line_bytes, bytes) else line_bytes
                line = line.strip()
                
                if not line:
                    continue  # Skip empty lines
                
                # Try to parse as JSON
                try:
                    response = json.loads(line)
                    # If we got valid JSON, check if it's a JSON-RPC response
                    if "jsonrpc" in response or "result" in response or "error" in response:
                        return response  # Found valid JSON-RPC response
                    else:
                        # Valid JSON but not JSON-RPC - continue reading
                        logger.debug(f"Received non-JSON-RPC JSON: {line[:100]}")
                        continue
                except json.JSONDecodeError:
                    # Not JSON - this might be a status message or prompt
                    # Log it but continue reading
                    logger.debug(f"Skipping non-JSON line from MCP server: {line[:100]}")
                    continue
            
            except asyncio.TimeoutError:
                if attempt == 0:
                    logger.warning(f"Timeout waiting for {operation} response")
                    return None
                break
        
        return response
    
    async def _send_initialize(self, server_id: str, state: ServerState) -> None:
        """Send initialize request to MCP server"""
        import json
        
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": self.default_client_name,
                    "version": self.default_client_version
                }
            }
        }
        
        try:
            # Check if process exists and has required streams
            if not state.process:
                raise Exception("Process not created - cannot initialize")
            
            if not state.process.stdin:
                raise Exception("Process stdin not available")
            
            if not state.process.stdout:
                raise Exception("Process stdout not available")
            
            # Initialize read lock if not already initialized
            if state._read_lock is None:
                state._read_lock = asyncio.Lock()
            
            request_str = json.dumps(initialize_request) + "\n"
            # Encode to bytes if needed
            if isinstance(request_str, str):
                request_bytes = request_str.encode('utf-8')
            else:
                request_bytes = request_str
            
            state.process.stdin.write(request_bytes)
            await state.process.stdin.drain()
            
            # Read initialize response
            # Some MCP servers output non-JSON text first (like status messages)
            # We need to skip non-JSON lines and read until we get valid JSON
            # Use lock to prevent concurrent reads
            async with state._read_lock:
                response = await self._read_jsonrpc_response(
                    state.process.stdout,
                    timeout=5.0,
                    operation="initialization"
                )
                
                if not response:
                    raise Exception("Could not find valid JSON-RPC response from MCP server after reading multiple lines")
                
                # Process the JSON-RPC response
                if "result" in response:
                    logger.info(f"Initialized MCP server: {server_id}")
                    # Send initialized notification
                    initialized_notification = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    }
                    notif_str = json.dumps(initialized_notification) + "\n"
                    notif_bytes = notif_str.encode('utf-8') if isinstance(notif_str, str) else notif_str
                    state.process.stdin.write(notif_bytes)
                    await state.process.stdin.drain()
                elif "error" in response:
                    error_info = response.get("error", {})
                    error_msg = error_info.get("message", str(error_info))
                    raise Exception(f"MCP server initialization error: {error_msg}")
                else:
                    raise Exception(f"Unexpected response format from server: {response}")
                    
        except asyncio.TimeoutError:
            raise Exception("Timeout waiting for initialization response from MCP server")
        except Exception as e:
            error_msg = f"Failed to initialize MCP server '{server_id}': {e}"
            logger.error(error_msg)
            # Re-raise to fail the connection
            raise Exception(error_msg)
    
    async def _connect_via_http(self, server_id: str, state: ServerState, config: Dict[str, Any]) -> None:
        """Connect via HTTP transport (SSE or Streamable)"""
        url = config.get("url")
        if not url:
            raise ValueError("HTTP config must include 'url'")
        
        if isinstance(url, str):
            url = urlparse(url)
        
        prefer_sse = config.get("prefer_sse", False)
        if isinstance(url, str):
            prefer_sse = prefer_sse or url.endswith("/sse")
        
        # Create HTTP session
        session = aiohttp.ClientSession()
        state.session = session
        state.transport = "sse" if prefer_sse else "streamable-http"
        
        try:
            # Test connection with a ping or initialize request
            # For now, we'll just mark as connected
            # In a full implementation, you'd send initialize request
            logger.info(f"Connected via HTTP to server: {server_id}")
            
        except Exception as e:
            await session.close()
            state.session = None
            raise Exception(f"Failed to connect via HTTP: {str(e)}")
    
    async def disconnect_server(self, server_id: str) -> None:
        """Disconnect from an MCP server"""
        state = self.server_states.get(server_id)
        if not state:
            return
        
        if state.status != ConnectionStatus.CONNECTED:
            return
        
        try:
            # Close STDIO process
            if state.process:
                # For asyncio subprocess, use terminate and wait
                if hasattr(state.process, 'terminate'):
                    state.process.terminate()
                    try:
                        await asyncio.wait_for(state.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        state.process.kill()
                        await state.process.wait()
                state.process = None
            
            # Close HTTP session
            if state.session:
                await state.session.close()
                state.session = None
            
            state.status = ConnectionStatus.DISCONNECTED
            state.client = None
            state.transport = None
            logger.info(f"Disconnected from MCP server: {server_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from server '{server_id}': {e}")
            state.status = ConnectionStatus.DISCONNECTED
    
    async def remove_server(self, server_id: str) -> None:
        """Remove a server (disconnect and remove from manager)"""
        if server_id in self.server_states:
            # Disconnect if connected
            state = self.server_states[server_id]
            if state.status == ConnectionStatus.CONNECTED:
                await self.disconnect_server(server_id)
            del self.server_states[server_id]
    
    async def _fetch_server_tools(self, server_id: str) -> None:
        """Fetch tools from a connected MCP server and cache them"""
        state = self.server_states.get(server_id)
        if not state or state.status != ConnectionStatus.CONNECTED:
            return
        
        try:
            # Try to fetch tools via JSON-RPC over STDIO or HTTP
            if state.transport == "stdio" and state.process:
                tools = await self._fetch_tools_via_stdio(state)
            elif state.transport in ["sse", "streamable-http"] and state.session:
                tools = await self._fetch_tools_via_http(state)
            else:
                tools = []
            
            state.tools_cache = tools
            logger.info(f"Fetched {len(tools)} tools from server: {server_id}")
        except Exception as e:
            logger.warning(f"Failed to fetch tools from server '{server_id}': {e}")
            state.tools_cache = []
    
    async def _fetch_tools_via_stdio(self, state: ServerState) -> List[Dict[str, Any]]:
        """Fetch tools via STDIO transport using JSON-RPC"""
        import json
        import asyncio
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        try:
            if not state.process or not state.process.stdin or not state.process.stdout:
                logger.error("STDIO process not properly initialized")
                return []
            
            # Send request
            request_str = json.dumps(request) + "\n"
            # Encode to bytes if needed
            if isinstance(request_str, str):
                request_bytes = request_str.encode('utf-8')
            else:
                request_bytes = request_str
            
            state.process.stdin.write(request_bytes)
            await state.process.stdin.drain()
            
            # Read response with timeout - use lock to prevent concurrent reads
            async with state._read_lock:
                response = await self._read_jsonrpc_response(
                    state.process.stdout,
                    timeout=5.0,
                    operation="tools/list"
                )
            
            if response:
                if "result" in response and "tools" in response["result"]:
                    tools = response["result"]["tools"]
                    logger.info(f"Successfully fetched {len(tools)} tools via STDIO")
                    return tools
                elif "error" in response:
                    logger.error(f"MCP server error: {response['error']}")
                    return []
            
            return []
                
        except Exception as e:
            logger.error(f"Error fetching tools via STDIO: {e}", exc_info=True)
            return []
    
    async def _fetch_tools_via_http(self, state: ServerState) -> List[Dict[str, Any]]:
        """Fetch tools via HTTP transport"""
        import json
        
        if not state.session:
            return []
        
        config = state.config
        url = config.get("url", "")
        if not url:
            return []
        
        # Ensure URL is a string
        if not isinstance(url, str):
            url = str(url)
        
        # Add /messages endpoint if not present
        if not url.endswith("/messages"):
            url = url.rstrip("/") + "/messages"
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        try:
            async with state.session.post(
                url,
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data and "tools" in data["result"]:
                        tools = data["result"]["tools"]
                        logger.info(f"Successfully fetched {len(tools)} tools via HTTP")
                        return tools
                    elif "error" in data:
                        logger.error(f"MCP server error: {data['error']}")
                        return []
                else:
                    logger.error(f"HTTP error {response.status} when fetching tools")
                    return []
        except Exception as e:
            logger.error(f"Error fetching tools via HTTP: {e}", exc_info=True)
            return []
    
    async def list_tools(self, server_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """List tools from an MCP server"""
        state = self._ensure_connected(server_id)
        
        # Refresh tools if requested or cache is empty
        if force_refresh or not state.tools_cache:
            await self._fetch_server_tools(server_id)
        
        return {"tools": state.tools_cache or []}
    
    async def get_tools(self, server_ids: Optional[List[str]] = None, force_refresh: bool = False) -> Dict[str, Any]:
        """Get tools from multiple servers"""
        if not server_ids:
            # Only get tools from connected servers
            server_ids = [
                sid for sid in self.list_servers()
                if self.get_connection_status(sid) == ConnectionStatus.CONNECTED.value
            ]
        
        all_tools = []
        for server_id in server_ids:
            try:
                # Only fetch from connected servers
                if self.get_connection_status(server_id) == ConnectionStatus.CONNECTED.value:
                    result = await self.list_tools(server_id, force_refresh=force_refresh)
                    tools = result.get("tools", [])
                    # Convert MCP tools to OpenAI format and tag with server ID
                    for tool in tools:
                        # Ensure tool is in OpenAI format
                        if not isinstance(tool, dict):
                            continue
                        
                        # Convert MCP tool format to OpenAI format
                        converted_tool = None
                        
                        # Check if already in OpenAI format
                        if "type" in tool and "function" in tool:
                            converted_tool = tool.copy()
                        elif "name" in tool:
                            # Convert from MCP format
                            input_schema = tool.get("inputSchema", {})
                            if not input_schema and "parameters" in tool:
                                input_schema = tool["parameters"]
                            
                            converted_tool = {
                                "type": "function",
                                "function": {
                                    "name": tool.get("name", ""),
                                    "description": tool.get("description", ""),
                                    "parameters": input_schema or {}
                                }
                            }
                        else:
                            # Skip invalid tool format
                            logger.warning(f"Invalid tool format from server '{server_id}': {tool}")
                            continue
                        
                        # Store server_id internally for execution routing
                        # But don't include it in the tool schema sent to LLM
                        # We'll use a separate mapping for tool -> server lookup
                        # Note: _server_id is removed in tool_service before sending to LLM
                        converted_tool["_server_id"] = server_id
                        all_tools.append(converted_tool)
            except Exception as e:
                logger.error(f"Error getting tools from server '{server_id}': {e}", exc_info=True)
        
        logger.info(f"Total tools collected: {len(all_tools)} from {len(server_ids)} servers")
        return {"tools": all_tools}
    
    async def execute_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on an MCP server"""
        import json
        import asyncio
        
        state = self._ensure_connected(server_id)
        
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            if state.transport == "stdio" and state.process:
                # STDIO transport
                if state.process.stdin and state.process.stdout:
                    request_str = json.dumps(request) + "\n"
                    # Encode to bytes if needed
                    if isinstance(request_str, str):
                        request_bytes = request_str.encode('utf-8')
                    else:
                        request_bytes = request_str
                    
                    state.process.stdin.write(request_bytes)
                    await state.process.stdin.drain()
                    
                    # Read response - use lock to prevent concurrent reads
                    async with state._read_lock:
                        response = await self._read_jsonrpc_response(
                            state.process.stdout,
                            timeout=30.0,
                            operation="tool execution"
                        )
                    
                    if response:
                        if "result" in response:
                            return response["result"]
                        elif "error" in response:
                            error_info = response.get("error", {})
                            error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                            raise Exception(f"MCP tool error: {error_msg}")
                        else:
                            raise Exception(f"Unexpected response format: {response}")
                    else:
                        raise Exception("Could not find valid JSON-RPC response from MCP server")
            
            elif state.transport in ["sse", "streamable-http"] and state.session:
                # HTTP transport
                config = state.config
                url = config.get("url", "")
                if not isinstance(url, str):
                    url = str(url)
                if not url.endswith("/messages"):
                    url = url.rstrip("/") + "/messages"
                
                async with state.session.post(
                    url,
                    json=request,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "result" in data:
                            return data["result"]
                        elif "error" in data:
                            raise Exception(f"MCP tool error: {data['error']}")
                    else:
                        raise Exception(f"HTTP error {response.status}")
            
            raise Exception("Tool execution not supported for this transport type")
            
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}' on server '{server_id}': {e}")
            raise
    
    async def list_resources(self, server_id: str) -> Dict[str, Any]:
        """List resources from an MCP server"""
        state = self._ensure_connected(server_id)
        
        if state.resources_cache:
            return {"resources": state.resources_cache}
        
        return {"resources": []}
    
    async def list_prompts(self, server_id: str) -> Dict[str, Any]:
        """List prompts from an MCP server"""
        state = self._ensure_connected(server_id)
        
        if state.prompts_cache:
            return {"prompts": state.prompts_cache}
        
        return {"prompts": []}
    
    def _ensure_connected(self, server_id: str) -> ServerState:
        """Ensure server is connected, raise error if not"""
        state = self.server_states.get(server_id)
        if not state:
            raise ValueError(f"Unknown MCP server: '{server_id}'")
        
        if state.status != ConnectionStatus.CONNECTED:
            raise ValueError(f"MCP server '{server_id}' is not connected (status: {state.status.value})")
        
        return state
    
    async def disconnect_all_servers(self) -> None:
        """Disconnect from all servers"""
        server_ids = list(self.server_states.keys())
        for server_id in server_ids:
            await self.disconnect_server(server_id)
    
    def get_server_config(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get server configuration"""
        state = self.server_states.get(server_id)
        return state.config if state else None

