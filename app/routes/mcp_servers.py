"""
MCP Server Management API Routes
Handles connecting, disconnecting, and managing MCP servers
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from ..models import (
    MCPServerConnectRequest,
    MCPServerConnectResponse,
    MCPServerListResponse,
    MCPServerSummary
)
from ..mcp_client_manager import MCPClientManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Global MCP client manager instance
mcp_manager: MCPClientManager = None


def set_mcp_manager(manager: MCPClientManager):
    """Set the MCP client manager instance"""
    global mcp_manager
    mcp_manager = manager


@router.get("/v1/mcp/servers", response_model=MCPServerListResponse)
async def list_mcp_servers():
    """List all MCP servers"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP manager not initialized")
    
    summaries = mcp_manager.get_server_summaries()
    return MCPServerListResponse(
        servers=[MCPServerSummary(**summary) for summary in summaries]
    )


@router.post("/v1/mcp/servers/connect", response_model=MCPServerConnectResponse)
async def connect_mcp_server(request: MCPServerConnectRequest):
    """Connect to an MCP server"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP manager not initialized")
    
    try:
        # Convert Pydantic model to dict
        config_dict = request.server_config.dict(exclude_none=True)
        
        await mcp_manager.connect_to_server(request.server_id, config_dict)
        
        # Verify connection status
        status = mcp_manager.get_connection_status(request.server_id)
        if status != "connected":
            raise Exception(f"Connection completed but server status is '{status}'")
        
        # Tools are automatically fetched during connection
        # Return success with tool count if available
        try:
            tools_result = await mcp_manager.list_tools(request.server_id)
            tool_count = len(tools_result.get("tools", []))
            status_msg = f"connected ({tool_count} tools)" if tool_count > 0 else "connected"
        except Exception as tool_error:
            # If tools can't be fetched, still consider it connected but note the issue
            logger.warning(f"Connected but failed to fetch tools: {tool_error}")
            status_msg = "connected (tools unavailable)"
        
        return MCPServerConnectResponse(
            success=True,
            status=status_msg
        )
    except ValueError as e:
        # Return 400 for client errors (like already connected)
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        # Return 500 for server/connection errors
        error_msg = f"Connection failed: {str(e)}"
        logger.error(f"MCP connection error: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@router.post("/v1/mcp/servers/{server_id}/disconnect")
async def disconnect_mcp_server(server_id: str):
    """Disconnect from an MCP server"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP manager not initialized")
    
    if not mcp_manager.has_server(server_id):
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")
    
    try:
        await mcp_manager.disconnect_server(server_id)
        return {"success": True, "status": "disconnected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disconnect: {str(e)}")


@router.delete("/v1/mcp/servers/{server_id}")
async def remove_mcp_server(server_id: str):
    """Remove an MCP server"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP manager not initialized")
    
    if not mcp_manager.has_server(server_id):
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")
    
    try:
        await mcp_manager.remove_server(server_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove server: {str(e)}")


@router.get("/v1/mcp/servers/{server_id}/status")
async def get_mcp_server_status(server_id: str):
    """Get status of an MCP server"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP manager not initialized")
    
    if not mcp_manager.has_server(server_id):
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")
    
    status = mcp_manager.get_connection_status(server_id)
    config = mcp_manager.get_server_config(server_id)
    
    return {
        "server_id": server_id,
        "status": status,
        "config": config
    }


@router.get("/v1/mcp/servers/{server_id}/tools")
async def get_mcp_server_tools(server_id: str):
    """Get tools from an MCP server"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP manager not initialized")
    
    if not mcp_manager.has_server(server_id):
        raise HTTPException(status_code=404, detail=f"MCP server '{server_id}' not found")
    
    try:
        result = await mcp_manager.list_tools(server_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {str(e)}")


@router.get("/v1/mcp/tools")
async def get_all_mcp_tools(server_ids: str = None):
    """Get tools from all MCP servers or specified servers"""
    if not mcp_manager:
        raise HTTPException(status_code=500, detail="MCP manager not initialized")
    
    try:
        server_id_list = None
        if server_ids:
            server_id_list = [s.strip() for s in server_ids.split(",")]
        
        result = await mcp_manager.get_tools(server_id_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {str(e)}")


@router.get("/v1/tools")
async def get_available_tools():
    """Get list of all available tools (built-in + MCP)"""
    from ..tool_service import tool_service
    
    try:
        tools = await tool_service.get_available_tools()
        return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {str(e)}")

