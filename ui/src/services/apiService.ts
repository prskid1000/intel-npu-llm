import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'tool' | 'system';
  content: string;
  audioData?: string;
  imageUrl?: string;
  videoUrl?: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

export interface ChatResponse {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
      audio?: {
        data: string;
        format: string;
      };
      tool_calls?: ToolCall[];
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  conversation_messages?: Array<{
    role: string;
    content: string;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
  }>;
}

export interface Tool {
  type: string;
  function: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  };
}

export interface ModelInfo {
  id: string;
  created: number;
  owned_by: string;
  permission: any[];
  root: string;
  parent: string | null;
  capabilities?: {
    type?: string;
  };
}

export interface ModelsResponse {
  data: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  models_loaded?: number;
  llm_models?: number;
  vlm_models?: number;
  [key: string]: any;
}

class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await axios.get<HealthResponse>(`${this.baseURL}/health`);
      return response.data;
    } catch (error) {
      throw new Error('Failed to connect to server');
    }
  }

  async getModels(): Promise<ModelsResponse> {
    try {
      const response = await axios.get<ModelsResponse>(`${this.baseURL}/v1/models`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch models');
    }
  }

  async sendMessage(
    text: string,
    audioFile?: File,
    imageFile?: File,
    videoFile?: File,
    options?: {
      model?: string;
      maxTokens?: number;
      temperature?: number;
      topP?: number;
      returnAudio?: boolean;
      tools?: Tool[];
      messages?: ChatMessage[];
      stream?: boolean;
    }
  ): Promise<ChatResponse> {
    // Build messages array from history
    const messages: any[] = [];
    
    // Process history messages (excluding the current one)
    if (options?.messages && options.messages.length > 0) {
      for (const msg of options.messages) {
        const apiMsg: any = {
          role: msg.role,
          content: msg.content || '',
        };

        // For user messages: include media inputs (imageUrl)
        if (msg.role === 'user') {
          if (msg.imageUrl) {
            // Convert data URL to base64 if needed
            apiMsg.content = [
              {
                type: 'image_url',
                image_url: {
                  url: msg.imageUrl
                }
              },
              {
                type: 'text',
                text: msg.content || ''
              }
            ];
          }
        }
        
        // Include tool calls and tool call IDs
        if (msg.toolCalls) {
          apiMsg.tool_calls = msg.toolCalls;
        }
        if (msg.toolCallId) {
          apiMsg.tool_call_id = msg.toolCallId;
        }

        messages.push(apiMsg);
      }
    }

    // Add the current message
    const currentMessage: any = {
      role: 'user',
    };

    // Handle audio file (voice input)
    if (audioFile) {
      const audioBase64 = await this.fileToBase64(audioFile);
      const mimeType = audioFile.type || 'audio/webm';
      currentMessage.content = [
        {
          type: 'audio',
          audio: {
            data: `data:${mimeType};base64,${audioBase64}`
          }
        },
        {
          type: 'text',
          text: text || ''
        }
      ];
    } else if (imageFile) {
      // Handle image file
      const imageBase64 = await this.fileToBase64(imageFile);
      const mimeType = imageFile.type || 'image/png';
      currentMessage.content = [
        {
          type: 'image_url',
          image_url: {
            url: `data:${mimeType};base64,${imageBase64}`
          }
        },
        {
          type: 'text',
          text: text || ''
        }
      ];
    } else {
      currentMessage.content = text || '';
    }

    messages.push(currentMessage);

    const requestBody: any = {
      model: options?.model || 'qwen2.5-3b',
      messages,
      max_tokens: options?.maxTokens || 512,
      temperature: options?.temperature || 0.7,
      top_p: options?.topP || 0.9,
      stream: options?.stream || false,
    };

    // Add tools if provided
    if (options?.tools && options.tools.length > 0) {
      requestBody.tools = options.tools;
    }

    // Add audio output if requested (via modalities)
    if (options?.returnAudio) {
      requestBody.modalities = ['audio'];
    }

    try {
      const response = await axios.post<ChatResponse>(
        `${this.baseURL}/v1/chat/completions`,
        requestBody,
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 300000,
        }
      );
      
      return response.data;
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data?.detail || 'Server error');
      } else if (error.request) {
        throw new Error('No response from server. Make sure the server is running.');
      } else {
        throw new Error(error.message || 'Request failed');
      }
    }
  }

  private async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Extract base64 from data URL
        if (result.startsWith('data:')) {
          resolve(result.split(',')[1]);
        } else {
          resolve(result);
        }
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  decodeAudioBase64(base64Data: string): string {
    // Convert base64 to data URL (persists across reloads)
    return `data:audio/wav;base64,${base64Data}`;
  }

  async getAvailableTools(): Promise<{ tools: Tool[] }> {
    try {
      const response = await axios.get<{ tools: Tool[] }>(`${this.baseURL}/v1/tools`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch available tools');
    }
  }

  // MCP Server Management APIs
  async getMCPServers(): Promise<{ servers: Array<{ id: string; status: string; config: any; error?: string }> }> {
    try {
      const response = await axios.get(`${this.baseURL}/v1/mcp/servers`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch MCP servers');
    }
  }

  async connectMCPServer(serverId: string, serverConfig: any): Promise<{ success: boolean; status: string; error?: string }> {
    try {
      const response = await axios.post(`${this.baseURL}/v1/mcp/servers/connect`, {
        server_id: serverId,
        server_config: serverConfig,
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to connect MCP server');
    }
  }

  async disconnectMCPServer(serverId: string): Promise<void> {
    try {
      await axios.post(`${this.baseURL}/v1/mcp/servers/${serverId}/disconnect`);
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to disconnect MCP server');
    }
  }

  async removeMCPServer(serverId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseURL}/v1/mcp/servers/${serverId}`);
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to remove MCP server');
    }
  }

  async getMCPServerStatus(serverId: string): Promise<{ server_id: string; status: string; config: any }> {
    try {
      const response = await axios.get(`${this.baseURL}/v1/mcp/servers/${serverId}/status`);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to get MCP server status');
    }
  }
}

export const apiService = new ApiService();

