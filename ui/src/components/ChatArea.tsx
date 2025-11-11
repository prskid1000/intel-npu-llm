import { useEffect, useRef, useMemo } from 'react';
import { MessageBubble } from './MessageBubble';
import { Loader2 } from 'lucide-react';
import { Chat } from '../utils/storage';
import { apiService } from '../services/apiService';

interface ChatAreaProps {
  chat: Chat | null;
  isLoading?: boolean;
}

export function ChatArea({ chat, isLoading = false }: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive in the same chat
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chat?.messages, chat?.id]);

  useEffect(() => {
    // Scroll to top when switching to a different chat
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = 0;
    }
  }, [chat?.id]);

  if (!chat) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-2">Welcome to NPU Chat</h2>
          <p className="text-dark-textSecondary">
            Start a new conversation to begin chatting with OpenVINO GenAI models
          </p>
        </div>
      </div>
    );
  }

  // Convert base64 audio data to data URLs (persists across reloads)
  const messagesWithMedia = useMemo(() => {
    const processedMessages = chat.messages
      .filter((message) => message.role !== 'system')
      .map((message) => {
        let audioUrl: string | undefined = message.audioData;
        
        // If audioData exists and is not already a data URL, convert from base64
        if (message.audioData && !message.audioData.startsWith('data:')) {
          try {
            if (message.audioData.length > 100 && !message.audioData.startsWith('blob:')) {
              audioUrl = apiService.decodeAudioBase64(message.audioData);
            } else if (message.audioData.startsWith('blob:')) {
              audioUrl = undefined;
            }
          } catch (error) {
            console.error('Error decoding audio:', error);
            audioUrl = undefined;
          }
        }
        
        let imageUrl = message.imageUrl;
        if (imageUrl && imageUrl.startsWith('blob:')) {
          imageUrl = undefined;
        }
        
        return {
          ...message,
          audioUrl,
          imageUrl,
        };
      });
    
    return processedMessages;
  }, [chat.messages]);

  return (
    <div ref={chatAreaRef} className="flex-1 overflow-y-auto scrollbar-thin p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {chat.messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-2xl font-semibold mb-2">{chat.title}</h2>
              <p className="text-dark-textSecondary">
                Send a message to start the conversation
              </p>
            </div>
          </div>
        ) : (
          <>
            {messagesWithMedia.map((message, index) => (
              <MessageBubble
                key={`${message.timestamp}-${index}`}
                role={message.role}
                content={message.content}
                audioUrl={message.audioUrl}
                imageUrl={message.imageUrl}
                timestamp={message.timestamp}
                toolCalls={message.toolCalls}
                toolCallId={message.toolCallId}
              />
            ))}
            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="bg-dark-surface rounded-2xl px-4 py-3 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-dark-textSecondary" />
                  <span className="text-dark-textSecondary">Thinking...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>
    </div>
  );
}

