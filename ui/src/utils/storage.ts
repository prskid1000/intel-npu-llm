export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

export interface Chat {
  id: string;
  title: string;
  messages: Array<{
    role: 'user' | 'assistant' | 'system' | 'tool';
    content: string;
    audioData?: string;
    imageUrl?: string;
    videoUrl?: string;
    timestamp: number;
    toolCalls?: ToolCall[];
    toolCallId?: string;
  }>;
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'npu_chat_history';
const MAX_CHATS = 50;

export const storage = {
  getChats(): Chat[] {
    try {
      const data = localStorage.getItem(STORAGE_KEY);
      if (!data) return [];
      return JSON.parse(data);
    } catch (error) {
      console.error('Error loading chats:', error);
      return [];
    }
  },

  saveChat(chat: Chat): void {
    try {
      const chats = this.getChats();
      const index = chats.findIndex(c => c.id === chat.id);
      
      if (index >= 0) {
        chats[index] = chat;
      } else {
        chats.unshift(chat);
      }
      
      // Keep only the most recent MAX_CHATS
      const sorted = chats.sort((a, b) => b.updatedAt - a.updatedAt);
      const limited = sorted.slice(0, MAX_CHATS);
      
      localStorage.setItem(STORAGE_KEY, JSON.stringify(limited));
    } catch (error) {
      console.error('Error saving chat:', error);
    }
  },

  getChat(id: string): Chat | null {
    const chats = this.getChats();
    return chats.find(c => c.id === id) || null;
  },

  deleteChat(id: string): void {
    try {
      const chats = this.getChats();
      const filtered = chats.filter(c => c.id !== id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  },

  clearAll(): void {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error('Error clearing chats:', error);
    }
  },

  generateChatId(): string {
    return `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },

  generateChatTitle(firstMessage: string): string {
    // Generate a title from the first message (first 50 chars)
    const title = firstMessage.trim().slice(0, 50);
    return title || 'New Chat';
  }
};

// MCP Server Storage
const MCP_SERVERS_STORAGE_KEY = 'npu_mcp_servers';

export interface MCPServerConfig {
  id: string;
  config: {
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    url?: string;
    prefer_sse?: boolean;
    timeout?: number;
    version?: string;
    [key: string]: any;
  };
  lastConnected?: number;
}

export const mcpStorage = {
  getServers(): MCPServerConfig[] {
    try {
      const data = localStorage.getItem(MCP_SERVERS_STORAGE_KEY);
      if (!data) return [];
      return JSON.parse(data);
    } catch (error) {
      console.error('Error loading MCP servers from storage:', error);
      return [];
    }
  },

  saveServer(server: MCPServerConfig): void {
    try {
      const servers = this.getServers();
      const index = servers.findIndex(s => s.id === server.id);
      
      if (index >= 0) {
        servers[index] = server;
      } else {
        servers.push(server);
      }
      
      localStorage.setItem(MCP_SERVERS_STORAGE_KEY, JSON.stringify(servers));
    } catch (error) {
      console.error('Error saving MCP server to storage:', error);
    }
  },

  getServer(id: string): MCPServerConfig | null {
    const servers = this.getServers();
    return servers.find(s => s.id === id) || null;
  },

  deleteServer(id: string): void {
    try {
      const servers = this.getServers();
      const filtered = servers.filter(s => s.id !== id);
      localStorage.setItem(MCP_SERVERS_STORAGE_KEY, JSON.stringify(filtered));
    } catch (error) {
      console.error('Error deleting MCP server from storage:', error);
    }
  },

  clearAll(): void {
    try {
      localStorage.removeItem(MCP_SERVERS_STORAGE_KEY);
    } catch (error) {
      console.error('Error clearing MCP servers from storage:', error);
    }
  },

  updateLastConnected(id: string): void {
    try {
      const servers = this.getServers();
      const server = servers.find(s => s.id === id);
      if (server) {
        server.lastConnected = Date.now();
        localStorage.setItem(MCP_SERVERS_STORAGE_KEY, JSON.stringify(servers));
      }
    } catch (error) {
      console.error('Error updating last connected time:', error);
    }
  }
};

