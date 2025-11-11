import { useState, useEffect } from 'react';
import { Server, Plus, Trash2, Power, PowerOff, AlertCircle, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { apiService } from '../services/apiService';
import { mcpStorage, MCPServerConfig } from '../utils/storage';

interface MCPServer {
  id: string;
  status: 'disconnected' | 'connecting' | 'connected';
  config: {
    command?: string;
    url?: string;
    [key: string]: any;
  };
  error?: string;
}

interface MCPServerManagerProps {
  isOpen?: boolean;
  onToggle?: () => void;
}

export function MCPServerManager({ isOpen: controlledOpen, onToggle }: MCPServerManagerProps) {
  const [internalOpen, setInternalOpen] = useState(false);
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newServerId, setNewServerId] = useState('');
  const [newServerConfig, setNewServerConfig] = useState({
    type: 'stdio' as 'stdio' | 'http',
    command: '',
    args: '',
    env: '',
    url: '',
    prefer_sse: false,
  });

  const isOpen = controlledOpen !== undefined ? controlledOpen : internalOpen;
  const toggle = onToggle || (() => setInternalOpen(!internalOpen));

  // Load servers on mount and sync with backend
  useEffect(() => {
    if (isOpen) {
      loadServers();
    }
  }, [isOpen]);

  // Initial sync: load from localStorage and sync with backend
  useEffect(() => {
    const sync = async () => {
      await syncServersWithBackend();
      // After sync, load servers to show updated state
      await loadServers();
    };
    sync();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const syncServersWithBackend = async () => {
    try {
      // Load servers from localStorage
      const localServers = mcpStorage.getServers();
      
      // Load servers from backend
      const backendResponse = await apiService.getMCPServers();
      const backendServers = backendResponse.servers || [];
      const backendServerIds = new Set(backendServers.map(s => s.id));
      
      // Add any local servers that don't exist on backend
      for (const localServer of localServers) {
        if (!backendServerIds.has(localServer.id)) {
          // Server exists locally but not on backend - try to add it
          try {
            await apiService.connectMCPServer(localServer.id, localServer.config);
          } catch (err) {
            console.warn(`Failed to sync server ${localServer.id} to backend:`, err);
          }
        }
      }
      
      // Update local storage with backend servers (to sync any changes)
      for (const backendServer of backendServers) {
        mcpStorage.saveServer({
          id: backendServer.id,
          config: backendServer.config,
        });
      }
    } catch (err) {
      console.error('Error syncing servers with backend:', err);
    }
  };

  const loadServers = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.getMCPServers();
      const backendServers = response.servers || [];
      
      // Merge with local storage (local storage is source of truth for configs)
      const localServers = mcpStorage.getServers();
      const localServerMap = new Map(localServers.map(s => [s.id, s]));
      
      // Merge: use backend status, but keep local configs
      const mergedServers = backendServers.map(backendServer => {
        const localServer = localServerMap.get(backendServer.id);
        return {
          ...backendServer,
          config: localServer?.config || backendServer.config,
        };
      });
      
      // Add any local servers that don't exist on backend yet
      for (const localServer of localServers) {
        if (!mergedServers.find(s => s.id === localServer.id)) {
          mergedServers.push({
            id: localServer.id,
            status: 'disconnected' as const,
            config: localServer.config,
          });
        }
      }
      
      setServers(mergedServers);
    } catch (err: any) {
      setError(err.message || 'Failed to load MCP servers');
      // Fallback to local storage if backend fails
      const localServers = mcpStorage.getServers();
      setServers(localServers.map(s => ({
        id: s.id,
        status: 'disconnected' as const,
        config: s.config,
      })));
    } finally {
      setLoading(false);
    }
  };

  const handleConnect = async (serverId: string) => {
    const server = servers.find(s => s.id === serverId);
    if (!server) return;

    setLoading(true);
    try {
      const config: any = { ...server.config };
      if (config.command) {
        // STDIO config
        if (config.args && typeof config.args === 'string') {
          config.args = config.args.split(',').map((a: string) => a.trim());
        }
      }

      const result = await apiService.connectMCPServer(serverId, config);
      if (result.success) {
        // Save to localStorage
        mcpStorage.saveServer({
          id: serverId,
          config: config,
          lastConnected: Date.now(),
        });
        mcpStorage.updateLastConnected(serverId);
        
        await loadServers();
        
        // Trigger tools refresh automatically after successful connection
        // Use a small delay to ensure backend has fully processed the connection
        setTimeout(() => {
          if (window.dispatchEvent) {
            window.dispatchEvent(new CustomEvent('mcpServerConnected', { detail: { serverId } }));
            window.dispatchEvent(new CustomEvent('refreshTools'));
          }
        }, 500);
      } else {
        setError(result.error || 'Connection failed');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to connect');
    } finally {
      setLoading(false);
    }
  };

  const handleDisconnect = async (serverId: string) => {
    setLoading(true);
    try {
      await apiService.disconnectMCPServer(serverId);
      await loadServers();
      // Trigger tools refresh in parent component
      if (window.dispatchEvent) {
        window.dispatchEvent(new CustomEvent('mcpServerDisconnected', { detail: { serverId } }));
      }
    } catch (err: any) {
      setError(err.message || 'Failed to disconnect');
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (serverId: string) => {
    if (!confirm(`Are you sure you want to remove server "${serverId}"?`)) {
      return;
    }

    setLoading(true);
    try {
      // Remove from backend
      try {
        await apiService.removeMCPServer(serverId);
      } catch (err) {
        // Continue even if backend removal fails
        console.warn('Failed to remove server from backend:', err);
      }
      
      // Remove from localStorage
      mcpStorage.deleteServer(serverId);
      
      await loadServers();
      
      // Trigger tools refresh
      if (window.dispatchEvent) {
        window.dispatchEvent(new CustomEvent('mcpServerDisconnected', { detail: { serverId } }));
      }
    } catch (err: any) {
      setError(err.message || 'Failed to remove server');
    } finally {
      setLoading(false);
    }
  };

  const handleAddServer = async () => {
    if (!newServerId.trim()) {
      setError('Server ID is required');
      return;
    }

    setLoading(true);
    try {
      const config: any = {};
      if (newServerConfig.type === 'stdio') {
        if (!newServerConfig.command) {
          setError('Command is required for STDIO servers');
          setLoading(false);
          return;
        }
        config.command = newServerConfig.command;
        if (newServerConfig.args) {
          config.args = newServerConfig.args.split(',').map(a => a.trim()).filter(Boolean);
        }
        if (newServerConfig.env) {
          try {
            config.env = JSON.parse(newServerConfig.env);
          } catch {
            setError('Invalid JSON in environment variables');
            setLoading(false);
            return;
          }
        }
      } else {
        if (!newServerConfig.url) {
          setError('URL is required for HTTP servers');
          setLoading(false);
          return;
        }
        config.url = newServerConfig.url;
        config.prefer_sse = newServerConfig.prefer_sse;
      }

      const result = await apiService.connectMCPServer(newServerId, config);
      if (result.success) {
        // Save to localStorage
        mcpStorage.saveServer({
          id: newServerId,
          config: config,
          lastConnected: Date.now(),
        });
        
        setShowAddModal(false);
        setNewServerId('');
        setNewServerConfig({
          type: 'stdio',
          command: '',
          args: '',
          env: '',
          url: '',
          prefer_sse: false,
        });
        await loadServers();
        
        // Trigger tools refresh automatically after successful connection
        // Use a small delay to ensure backend has fully processed the connection
        setTimeout(() => {
          if (window.dispatchEvent) {
            window.dispatchEvent(new CustomEvent('mcpServerConnected', { detail: { serverId: newServerId } }));
            window.dispatchEvent(new CustomEvent('refreshTools'));
          }
        }, 500);
      } else {
        setError(result.error || 'Failed to add server');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to add server');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'text-green-500';
      case 'connecting':
        return 'text-yellow-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="border-t border-dark-border">
      <button
        onClick={toggle}
        className="w-full flex items-center justify-between p-4 hover:bg-dark-surfaceHover transition-colors"
      >
        <div className="flex items-center gap-2">
          <Server className="w-4 h-4 text-dark-accent" />
          <span className="text-sm font-medium text-dark-text">
            MCP Servers {servers.length > 0 && `(${servers.length})`}
          </span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-4 h-4 text-dark-textSecondary" />
        ) : (
          <ChevronDown className="w-4 h-4 text-dark-textSecondary" />
        )}
      </button>

      {isOpen && (
        <div className="px-4 pb-4 space-y-3 max-h-96 overflow-y-auto">
          {error && (
            <div className="flex items-center gap-2 p-3 bg-dark-error/20 border border-dark-error/50 rounded-lg">
              <AlertCircle className="w-4 h-4 text-dark-error flex-shrink-0" />
              <div className="flex-1">
                <div className="text-xs font-medium text-dark-error mb-1">Error</div>
                <div className="text-xs text-dark-textSecondary">{error}</div>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-dark-error hover:text-dark-error/80"
              >
                ×
              </button>
            </div>
          )}

          <button
            onClick={() => setShowAddModal(true)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-dark-accent hover:bg-dark-accentHover rounded-lg transition-colors text-sm"
          >
            <Plus className="w-4 h-4" />
            Add MCP Server
          </button>

          {loading && !servers.length ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-4 h-4 animate-spin text-dark-accent mr-2" />
              <span className="text-xs text-dark-textSecondary">Loading servers...</span>
            </div>
          ) : servers.length === 0 ? (
            <div className="text-center py-4">
              <div className="text-xs text-dark-textSecondary">No MCP servers configured</div>
            </div>
          ) : (
            <div className="space-y-2">
              {servers.map((server) => (
                <div
                  key={server.id}
                  className="bg-dark-surfaceSecondary rounded-lg p-3 border border-dark-border"
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-sm text-dark-text mb-1 truncate">
                        {server.id}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs ${getStatusColor(server.status)}`}>
                          {server.status}
                        </span>
                        {server.config.command && (
                          <span className="text-xs text-dark-textSecondary">
                            STDIO: {server.config.command}
                          </span>
                        )}
                        {server.config.url && (
                          <span className="text-xs text-dark-textSecondary truncate">
                            HTTP: {server.config.url}
                          </span>
                        )}
                      </div>
                      {server.error && (
                        <div className="text-xs text-dark-error mt-1">{server.error}</div>
                      )}
                    </div>
                    <div className="flex items-center gap-1">
                      {server.status === 'connected' ? (
                        <button
                          onClick={() => handleDisconnect(server.id)}
                          className="p-1 hover:bg-dark-surface rounded transition-colors text-green-500 hover:text-green-400"
                          title="Disconnect"
                        >
                          <PowerOff className="w-4 h-4" />
                        </button>
                      ) : (
                        <button
                          onClick={() => handleConnect(server.id)}
                          className="p-1 hover:bg-dark-surface rounded transition-colors text-gray-500 hover:text-dark-accent"
                          title="Connect"
                        >
                          <Power className="w-4 h-4" />
                        </button>
                      )}
                      <button
                        onClick={() => handleRemove(server.id)}
                        className="p-1 hover:bg-dark-surface rounded transition-colors text-dark-textSecondary hover:text-dark-error"
                        title="Remove"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Add Server Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-dark-surface border border-dark-border rounded-lg p-6 max-w-md w-full max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-dark-text mb-4">Add MCP Server</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-dark-text mb-1">
                  Server ID
                </label>
                <input
                  type="text"
                  value={newServerId}
                  onChange={(e) => setNewServerId(e.target.value)}
                  className="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded-lg focus:outline-none focus:ring-2 focus:ring-dark-accent text-sm"
                  placeholder="my-mcp-server"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-dark-text mb-1">
                  Transport Type
                </label>
                <select
                  value={newServerConfig.type}
                  onChange={(e) => setNewServerConfig({ ...newServerConfig, type: e.target.value as 'stdio' | 'http' })}
                  className="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded-lg focus:outline-none focus:ring-2 focus:ring-dark-accent text-sm"
                >
                  <option value="stdio">STDIO</option>
                  <option value="http">HTTP</option>
                </select>
              </div>

              {newServerConfig.type === 'stdio' ? (
                <>
                  <div>
                    <label className="block text-sm font-medium text-dark-text mb-1">
                      Command *
                    </label>
                    <input
                      type="text"
                      value={newServerConfig.command}
                      onChange={(e) => setNewServerConfig({ ...newServerConfig, command: e.target.value })}
                      className="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded-lg focus:outline-none focus:ring-2 focus:ring-dark-accent text-sm"
                      placeholder="node"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-dark-text mb-1">
                      Arguments (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={newServerConfig.args}
                      onChange={(e) => setNewServerConfig({ ...newServerConfig, args: e.target.value })}
                      className="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded-lg focus:outline-none focus:ring-2 focus:ring-dark-accent text-sm"
                      placeholder="server.js, --port, 3000"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-dark-text mb-1">
                      Environment Variables (JSON)
                    </label>
                    <textarea
                      value={newServerConfig.env}
                      onChange={(e) => setNewServerConfig({ ...newServerConfig, env: e.target.value })}
                      className="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded-lg focus:outline-none focus:ring-2 focus:ring-dark-accent text-sm font-mono"
                      rows={3}
                      placeholder='{"API_KEY": "value"}'
                    />
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <label className="block text-sm font-medium text-dark-text mb-1">
                      URL *
                    </label>
                    <input
                      type="text"
                      value={newServerConfig.url}
                      onChange={(e) => setNewServerConfig({ ...newServerConfig, url: e.target.value })}
                      className="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded-lg focus:outline-none focus:ring-2 focus:ring-dark-accent text-sm"
                      placeholder="https://api.example.com/mcp"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="prefer_sse"
                      checked={newServerConfig.prefer_sse}
                      onChange={(e) => setNewServerConfig({ ...newServerConfig, prefer_sse: e.target.checked })}
                      className="w-4 h-4"
                    />
                    <label htmlFor="prefer_sse" className="text-sm text-dark-text">
                      Prefer Server-Sent Events (SSE)
                    </label>
                  </div>
                </>
              )}

              <div className="flex gap-2 pt-2">
                <button
                  onClick={handleAddServer}
                  disabled={loading}
                  className="flex-1 px-4 py-2 bg-dark-accent hover:bg-dark-accentHover rounded-lg transition-colors text-sm font-medium disabled:opacity-50"
                >
                  {loading ? 'Adding...' : 'Add Server'}
                </button>
                <button
                  onClick={() => {
                    setShowAddModal(false);
                    setError(null);
                  }}
                  className="px-4 py-2 bg-dark-surfaceSecondary hover:bg-dark-surfaceHover rounded-lg transition-colors text-sm"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

