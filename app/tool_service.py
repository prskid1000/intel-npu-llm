"""
Tool Management Service
Handles tool registration, discovery, and execution
"""

from typing import Dict, Any, List, Optional, Callable
from .tool_executor import ToolExecutor, tool_executor
from .mcp_client_manager import MCPClientManager


class ToolService:
    """Service for managing tools"""
    
    def __init__(self, executor: Optional[ToolExecutor] = None, mcp_manager: Optional[MCPClientManager] = None):
        """
        Initialize tool service
        
        Args:
            executor: Optional ToolExecutor instance. If None, uses global instance.
            mcp_manager: Optional MCPClientManager instance for MCP tools.
        """
        self.executor = executor or tool_executor
        self.mcp_manager = mcp_manager
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of all available tools with their schemas
        Includes both built-in tools and MCP server tools
        
        Returns:
            List of tool schemas in OpenAI-compatible format
        """
        tools = self.executor.get_tool_schemas()
        
        # Add MCP tools if manager is available
        if self.mcp_manager:
            try:
                mcp_result = await self.mcp_manager.get_tools(force_refresh=True)
                mcp_tools = mcp_result.get("tools", [])
                
                # Log for debugging
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Found {len(mcp_tools)} MCP tools from {len(self.mcp_manager.list_servers())} servers")
                
                # Convert MCP tools to OpenAI format if needed
                for tool in mcp_tools:
                    # Ensure tool is in correct format
                    if isinstance(tool, dict):
                        # Remove internal metadata (like _server_id) before sending to LLM
                        clean_tool = {k: v for k, v in tool.items() if not k.startswith("_")}
                        
                        # If it's already in OpenAI format, use it
                        if "type" not in clean_tool:
                            clean_tool["type"] = "function"
                        if "function" not in clean_tool:
                            # Try to convert from MCP format
                            if "name" in clean_tool:
                                clean_tool = {
                                    "type": "function",
                                    "function": {
                                        "name": clean_tool.get("name", ""),
                                        "description": clean_tool.get("description", ""),
                                        "parameters": clean_tool.get("inputSchema", clean_tool.get("parameters", {}))
                                    }
                                }
                        
                        # Store server_id separately for execution routing (not in tool schema)
                        # We'll look it up during execution
                        tools.append(clean_tool)
            except Exception as e:
                # Log error but don't fail
                import logging
                logging.getLogger(__name__).warning(f"Failed to load MCP tools: {e}", exc_info=True)
        
        return tools
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool with given arguments
        Checks both built-in tools and MCP server tools
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool
            
        Returns:
            Result of tool execution
            
        Raises:
            ValueError: If tool is not found
            Exception: If tool execution fails
        """
        # Check if it's a built-in tool first
        if tool_name in self.executor.tool_registry:
            return self.executor.execute_tool(tool_name, arguments)
        
        # Check MCP tools - find which server has this tool
        if self.mcp_manager:
            # Create a mapping of tool names to server IDs for faster lookup
            # This avoids checking all servers on every execution
            tool_to_server = {}
            for server_id in self.mcp_manager.list_servers():
                if self.mcp_manager.get_connection_status(server_id) == "connected":
                    try:
                        tools_result = await self.mcp_manager.list_tools(server_id)
                        tools = tools_result.get("tools", [])
                        for tool in tools:
                            # Handle both MCP format and OpenAI format
                            tool_name_mcp = tool.get("name", "")
                            if not tool_name_mcp and "function" in tool:
                                tool_name_mcp = tool["function"].get("name", "")
                            
                            if tool_name_mcp and tool_name_mcp not in tool_to_server:
                                tool_to_server[tool_name_mcp] = server_id
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).debug(f"Error checking server {server_id}: {e}")
                        continue
            
            # Execute if found
            if tool_name in tool_to_server:
                server_id = tool_to_server[tool_name]
                return await self.mcp_manager.execute_tool(server_id, tool_name, arguments)
        
        raise ValueError(f"Tool '{tool_name}' not found")
    
    def register_tool(self, name: str, func: Callable, description: str = "", parameters: Optional[Dict[str, Any]] = None):
        """
        Register a new tool dynamically
        
        Args:
            name: Tool name
            func: Tool function to execute
            description: Tool description (optional, for schema)
            parameters: Tool parameters schema (optional)
        """
        self.executor.register_tool(name, func)
        # Note: If you want to update schemas dynamically, you'd need to extend ToolExecutor
    
    def is_tool_available(self, tool_name: str) -> bool:
        """
        Check if a tool is available
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is available, False otherwise
        """
        return tool_name in self.executor.tool_registry
    
    def get_tool_count(self) -> int:
        """
        Get the number of available tools
        
        Returns:
            Number of registered tools
        """
        return len(self.executor.tool_registry)


# Global tool service instance
tool_service = ToolService()

