import { useState, useEffect, useRef, useCallback } from 'react';
import { Menu, X, MessageSquare, Mic } from 'lucide-react';
import { ChatArea } from './ChatArea';
import { MessageInput } from './MessageInput';
import { VoiceModeInput } from './VoiceModeInput';
import { ChatSidebar } from './ChatSidebar';
import { useChatHistoryContext } from '../contexts/ChatHistoryContext';
import { apiService, ChatMessage, Tool } from '../services/apiService';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';

type InteractionMode = 'chat' | 'voice';

// Helper function to convert File to base64 data URL
function convertFileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function ChatContainer() {
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState<'online' | 'offline'>('online');
  const [audioOutputEnabled, setAudioOutputEnabled] = useState(false);
  const [toolCallingEnabled, setToolCallingEnabled] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('qwen2.5-3b');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [availableTools, setAvailableTools] = useState<Tool[]>([]);
  const [toolsLoading, setToolsLoading] = useState(true);
  const [toolsError, setToolsError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mode, setMode] = useState<InteractionMode>('chat');
  const lastAudioRef = useRef<HTMLAudioElement | null>(null);
  
  const {
    currentChat,
    createNewChat,
    addMessage,
  } = useChatHistoryContext();

  // Voice recorder for voice mode
  const {
    isRecording,
    audioBlob,
    error: recordingError,
    startRecording,
    stopRecording,
    clearRecording,
    getAudioFile,
  } = useVoiceRecorder();

  // Load available models on mount (only LLM and VLM models)
  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await apiService.getModels();
        // Filter to only show LLM and VLM models
        const chatModels = response.data.filter(m => {
          const modelType = m.capabilities?.type || 'unknown';
          return modelType === 'llm' || modelType === 'vlm';
        });
        const modelIds = chatModels.map(m => m.id);
        setAvailableModels(modelIds);
        if (modelIds.length > 0 && !modelIds.includes(selectedModel)) {
          setSelectedModel(modelIds[0]);
        }
      } catch (error) {
        console.error('Failed to load models:', error);
      }
    };
    loadModels();
  }, []);

  // Load available tools on mount
  const loadTools = useCallback(async () => {
    setToolsLoading(true);
    setToolsError(null);
    try {
      const response = await apiService.getAvailableTools();
      setAvailableTools(response.tools || []);
      if (!response.tools || response.tools.length === 0) {
        setToolsError('No tools available');
      }
    } catch (error: any) {
      console.error('Failed to load tools:', error);
      setToolsError(error.message || 'Failed to load tools');
      setAvailableTools([]);
    } finally {
      setToolsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTools();
  }, [loadTools]);

  // Refresh tools when MCP servers connect/disconnect or manual refresh
  useEffect(() => {
    const handleMCPChange = () => {
      loadTools();
    };
    
    const handleRefresh = () => {
      loadTools();
    };

    window.addEventListener('mcpServerConnected', handleMCPChange);
    window.addEventListener('mcpServerDisconnected', handleMCPChange);
    window.addEventListener('refreshTools', handleRefresh);

    return () => {
      window.removeEventListener('mcpServerConnected', handleMCPChange);
      window.removeEventListener('mcpServerDisconnected', handleMCPChange);
      window.removeEventListener('refreshTools', handleRefresh);
    };
  }, [loadTools]);

  // Check server health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiService.checkHealth();
        setServerStatus('online');
      } catch (error) {
        setServerStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const playAudioResponse = useCallback((audioUrl: string) => {
    // Stop previous audio if playing
    if (lastAudioRef.current) {
      lastAudioRef.current.pause();
      lastAudioRef.current = null;
    }

    const audio = new Audio(audioUrl);
    audio.play().catch(err => {
      console.error('Error playing audio:', err);
    });
    lastAudioRef.current = audio;
  }, []);

  const handleVoiceSend = useCallback(async (imageFile?: File, videoFile?: File) => {
    // In voice mode, we need either audio or media
    if (!audioBlob && !imageFile && !videoFile) return;

    const audioFile = audioBlob ? getAudioFile() : undefined;

    // Create a new chat if none exists
    let chatId = currentChat?.id;
    if (!chatId) {
      chatId = createNewChat();
    }

    setIsLoading(true);

    try {
      // Convert image/video files to base64 data URLs for persistence
      let imageDataUrl: string | undefined;
      let videoDataUrl: string | undefined;

      if (imageFile) {
        imageDataUrl = await convertFileToDataUrl(imageFile);
      }
      if (videoFile) {
        videoDataUrl = await convertFileToDataUrl(videoFile);
      }

      // Add user message
      const userMessage = {
        role: 'user' as const,
        content: audioFile ? '[Voice message]' : '[Media message]',
        timestamp: Date.now(),
        imageUrl: imageDataUrl,
        videoUrl: videoDataUrl,
      };
      addMessage(chatId, userMessage);

      // Update server status
      if (serverStatus === 'offline') {
        setServerStatus('online');
      }

      // Send to API, force audio response in voice mode
      const response = await apiService.sendMessage(
        '', // No text in voice mode
        audioFile || undefined,
        imageFile,
        videoFile,
        {
          model: selectedModel,
          returnAudio: true, // Always return audio in voice mode
          tools: toolCallingEnabled && availableTools.length > 0 ? availableTools : undefined,
        }
      );

      // Extract response
      const assistantMessage = response.choices[0]?.message;
      if (!assistantMessage) {
        throw new Error('No response from server');
      }

      // Store base64 audio data if present
      let audioData: string | undefined;
      if (assistantMessage.audio?.data) {
        audioData = assistantMessage.audio.data;
      }

      // Add assistant message
      const assistantMsg = {
        role: 'assistant' as const,
        content: assistantMessage.content,
        audioData: audioData,
        timestamp: Date.now(),
      };

      addMessage(chatId, assistantMsg);

      // Clear recording
      clearRecording();
    } catch (error: any) {
      if (error.message?.includes('connect') || error.message?.includes('No response')) {
        setServerStatus('offline');
      }

      const errorMsg = {
        role: 'assistant' as const,
        content: `Error: ${error.message || 'Failed to get response from server'}`,
        timestamp: Date.now(),
      };
      addMessage(chatId, errorMsg);
      if (audioBlob) {
        clearRecording();
      }
    } finally {
      setIsLoading(false);
    }
  }, [audioBlob, getAudioFile, currentChat?.id, addMessage, createNewChat, serverStatus, selectedModel, toolCallingEnabled, availableTools, clearRecording]);

  const handleSend = async (
    text: string,
    audioFile?: File,
    imageFile?: File,
    videoFile?: File
  ) => {
    // Check if we have content to send
    if (!text.trim() && !imageFile && !videoFile && !audioFile) {
      return;
    }

    // Create a new chat if none exists
    let chatId = currentChat?.id;
    if (!chatId) {
      chatId = createNewChat();
    }

    setIsLoading(true);

    try {
      // Convert image/video files to base64 data URLs for persistence
      let imageDataUrl: string | undefined;
      let videoDataUrl: string | undefined;

      if (imageFile) {
        imageDataUrl = await convertFileToDataUrl(imageFile);
      }
      if (videoFile) {
        videoDataUrl = await convertFileToDataUrl(videoFile);
      }

      // Add user message to chat
      const userMessage = {
        role: 'user' as const,
        content: text || '[Media message]',
        timestamp: Date.now(),
        imageUrl: imageDataUrl,
        videoUrl: videoDataUrl,
        audioData: audioFile ? await convertFileToDataUrl(audioFile) : undefined,
      };

      addMessage(chatId, userMessage);

      // Update server status to online when sending
      if (serverStatus === 'offline') {
        setServerStatus('online');
      }

      // Build messages array for chat history
      const messages: ChatMessage[] = currentChat?.messages.map(msg => {
        const historyMsg: ChatMessage = {
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
          toolCalls: msg.toolCalls,
          toolCallId: msg.toolCallId,
        };
        
        // For user messages: include all media inputs
        if (msg.role === 'user') {
          historyMsg.imageUrl = msg.imageUrl;
          historyMsg.videoUrl = msg.videoUrl;
          historyMsg.audioData = msg.audioData;
        }
        
        return historyMsg;
      }) || [];

      // Add the new user message
      messages.push({
        role: 'user',
        content: text || '[Media message]',
        timestamp: Date.now(),
        imageUrl: imageDataUrl,
        videoUrl: videoDataUrl,
        audioData: audioFile ? await convertFileToDataUrl(audioFile) : undefined,
      });

      // Send to API
      const response = await apiService.sendMessage(
        text || 'Describe the uploaded media.',
        audioFile,
        imageFile,
        videoFile,
        {
          model: selectedModel,
          returnAudio: audioOutputEnabled,
          tools: toolCallingEnabled && availableTools.length > 0 ? availableTools : undefined,
          messages: messages.length > 0 ? messages : undefined,
        }
      );

      // Extract response
      const assistantMessage = response.choices[0]?.message;
      if (!assistantMessage) {
        throw new Error('No response from server');
      }

      // Store base64 audio data if present
      const audioData = assistantMessage.audio?.data;

      // Only add the NEW assistant response, not all conversation history
      // The conversation_messages field is only for tool calls/results, not for normal chat
      const assistantMsg: ChatMessage = {
        role: 'assistant' as const,
        content: assistantMessage.content,
        audioData: audioData,
        timestamp: Date.now(),
        toolCalls: assistantMessage.tool_calls,
      };
      addMessage(chatId, assistantMsg);
      
      // If conversation_messages are provided and contain tool-related messages, add those too
      // But only if they're not already in the chat (to avoid duplicates)
      if (response.conversation_messages && response.conversation_messages.length > 0) {
        for (const msg of response.conversation_messages) {
          // Only add tool messages and assistant messages with tool calls that aren't the final response
          if (msg.role === 'tool' || (msg.role === 'assistant' && msg.tool_calls && msg.content !== assistantMessage.content)) {
            const conversationMsg: ChatMessage = {
              role: msg.role as 'assistant' | 'tool',
              content: msg.content,
              timestamp: Date.now(),
              toolCalls: msg.tool_calls,
              toolCallId: msg.tool_call_id,
            };
            addMessage(chatId, conversationMsg);
          }
        }
      }

      // Auto-play audio if enabled
      if (audioData && audioOutputEnabled) {
        const audioUrl = apiService.decodeAudioBase64(audioData);
        playAudioResponse(audioUrl);
      }
    } catch (error: any) {
      // Check if it's a connection error and update server status
      if (error.message?.includes('connect') || error.message?.includes('No response')) {
        setServerStatus('offline');
      }

      // Add error message
      const errorMsg = {
        role: 'assistant' as const,
        content: `Error: ${error.message || 'Failed to get response from server'}`,
        timestamp: Date.now(),
      };
      addMessage(chatId, errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-dark-bg">
      {/* Sidebar */}
      <ChatSidebar 
        isOpen={sidebarOpen} 
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        availableTools={availableTools}
        toolsLoading={toolsLoading}
        toolsError={toolsError}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header with menu button */}
        <div className="flex items-center justify-between gap-4 p-4 border-b border-dark-border">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-dark-surfaceHover rounded-lg transition-colors md:hidden"
              aria-label="Toggle sidebar"
            >
              {sidebarOpen ? (
                <X className="w-5 h-5 text-dark-text" />
              ) : (
                <Menu className="w-5 h-5 text-dark-text" />
              )}
            </button>
            <h1 className="text-lg font-semibold text-dark-text">NPU Chat</h1>
            {availableModels.length > 0 && (
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="px-3 py-1.5 bg-dark-surface border border-dark-border rounded-lg text-dark-text text-sm focus:outline-none focus:ring-2 focus:ring-dark-accent"
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            )}
          </div>

          {/* Mode Toggle */}
          <div className="flex items-center gap-2 bg-dark-surface rounded-lg p-1">
            <button
              onClick={() => setMode('chat')}
              className={`px-4 py-2 rounded-md transition-colors flex items-center gap-2 ${
                mode === 'chat'
                  ? 'bg-dark-accent text-white'
                  : 'text-dark-textSecondary hover:text-dark-text'
              }`}
              aria-label="Chat mode"
            >
              <MessageSquare className="w-4 h-4" />
              <span className="text-sm font-medium">Chat</span>
            </button>
            <button
              onClick={() => {
                setMode('voice');
                clearRecording(); // Clear any existing recording when switching
              }}
              className={`px-4 py-2 rounded-md transition-colors flex items-center gap-2 ${
                mode === 'voice'
                  ? 'bg-dark-accent text-white'
                  : 'text-dark-textSecondary hover:text-dark-text'
              }`}
              aria-label="Voice mode"
            >
              <Mic className="w-4 h-4" />
              <span className="text-sm font-medium">Voice</span>
            </button>
          </div>
        </div>

        {/* Server status bar */}
        {serverStatus === 'offline' && (
          <div className="bg-dark-error/20 border-b border-dark-error/50 px-4 py-2">
            <p className="text-sm text-dark-error text-center">
              ⚠️ Server offline. Make sure the NPU server is running on port 8000
            </p>
          </div>
        )}

        {/* Chat area */}
        <ChatArea key={currentChat?.id || 'no-chat'} chat={currentChat} isLoading={isLoading} />

        {/* Input area */}
        {mode === 'chat' ? (
          <MessageInput
            onSend={handleSend}
            isLoading={isLoading}
            disabled={serverStatus === 'offline'}
            audioOutputEnabled={audioOutputEnabled}
            onAudioOutputToggle={() => setAudioOutputEnabled(!audioOutputEnabled)}
            toolCallingEnabled={toolCallingEnabled}
            onToolCallingToggle={() => setToolCallingEnabled(!toolCallingEnabled)}
          />
        ) : (
          <VoiceModeInput
            isRecording={isRecording}
            isLoading={isLoading}
            disabled={serverStatus === 'offline'}
            recordingError={recordingError}
            onStartRecording={startRecording}
            onStopRecording={stopRecording}
            onClearRecording={clearRecording}
            onSendWithMedia={handleVoiceSend}
          />
        )}
      </div>
    </div>
  );
}

