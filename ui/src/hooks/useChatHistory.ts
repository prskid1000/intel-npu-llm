import { useState, useEffect, useCallback, useMemo } from 'react';
import { storage, Chat } from '../utils/storage';

export function useChatHistory() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Load chats from storage on mount
  useEffect(() => {
    const loadedChats = storage.getChats();
    setChats(loadedChats);
    
    // Set the most recent chat as current if no current chat is set
    if (loadedChats.length > 0) {
      setCurrentChatId(prev => {
        // Only set if not already set
        if (prev === null) {
          const mostRecent = loadedChats.sort((a, b) => b.updatedAt - a.updatedAt)[0];
          return mostRecent.id;
        }
        return prev;
      });
    }
  }, []);

  // Listen for storage changes (for cross-tab synchronization)
  useEffect(() => {
    const handleStorageChange = () => {
      const loadedChats = storage.getChats();
      setChats(loadedChats);
      
      // If current chat was deleted, clear it or set to most recent
      if (currentChatId && !loadedChats.find(c => c.id === currentChatId)) {
        if (loadedChats.length > 0) {
          const mostRecent = loadedChats.sort((a, b) => b.updatedAt - a.updatedAt)[0];
          setCurrentChatId(mostRecent.id);
        } else {
          setCurrentChatId(null);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [currentChatId]);

  // Get current chat object
  const currentChat = useMemo(() => {
    if (!currentChatId) return null;
    return chats.find(c => c.id === currentChatId) || null;
  }, [chats, currentChatId]);

  // Filtered chats based on search query
  const filteredChats = useMemo(() => {
    if (!searchQuery.trim()) {
      return chats.sort((a, b) => b.updatedAt - a.updatedAt);
    }
    
    const query = searchQuery.toLowerCase();
    return chats
      .filter(chat => 
        chat.title.toLowerCase().includes(query) ||
        chat.messages.some(msg => msg.content.toLowerCase().includes(query))
      )
      .sort((a, b) => b.updatedAt - a.updatedAt);
  }, [chats, searchQuery]);

  // Create a new chat
  const createNewChat = useCallback((): string => {
    const newChat: Chat = {
      id: storage.generateChatId(),
      title: 'New Chat',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    
    // Save to storage
    storage.saveChat(newChat);
    
    // Update local state
    const updatedChats = storage.getChats();
    setChats(updatedChats);
    
    // Set as current chat
    setCurrentChatId(newChat.id);
    
    return newChat.id;
  }, []);

  // Add a message to a chat
  const addMessage = useCallback((chatId: string, message: Chat['messages'][0]) => {
    // Get chat from storage (source of truth)
    const chat = storage.getChat(chatId);
    if (!chat) {
      console.error(`Chat ${chatId} not found`);
      return;
    }

    // Create updated chat
    const updatedChat: Chat = {
      ...chat,
      messages: [...chat.messages, message],
      updatedAt: Date.now(),
    };

    // Auto-generate title from first user message
    if (updatedChat.messages.length === 1 && updatedChat.title === 'New Chat') {
      if (message.role === 'user') {
        updatedChat.title = storage.generateChatTitle(message.content);
      }
    }

    // Save to storage
    storage.saveChat(updatedChat);

    // Update local state
    const updatedChats = storage.getChats();
    setChats(updatedChats);
    
    // Update current chat if it's the one being modified
    if (currentChatId === chatId) {
      setCurrentChatId(chatId); // Trigger re-render
    }
  }, [currentChatId]);

  // Delete a chat
  const deleteChat = useCallback((chatId: string) => {
    storage.deleteChat(chatId);
    
    // Update local state
    const updatedChats = storage.getChats();
    setChats(updatedChats);
    
    // If deleted chat was current, switch to most recent or clear
    if (currentChatId === chatId) {
      if (updatedChats.length > 0) {
        const mostRecent = updatedChats.sort((a, b) => b.updatedAt - a.updatedAt)[0];
        setCurrentChatId(mostRecent.id);
      } else {
        setCurrentChatId(null);
      }
    }
  }, [currentChatId]);

  // Rename a chat
  const renameChat = useCallback((chatId: string, newTitle: string) => {
    const chat = storage.getChat(chatId);
    if (!chat) {
      console.error(`Chat ${chatId} not found`);
      return;
    }

    const updatedChat: Chat = {
      ...chat,
      title: newTitle.trim() || 'New Chat',
      updatedAt: Date.now(),
    };

    storage.saveChat(updatedChat);
    
    // Update local state
    const updatedChats = storage.getChats();
    setChats(updatedChats);
  }, []);

  // Clear all chats
  const clearAllChats = useCallback(() => {
    storage.clearAll();
    setChats([]);
    setCurrentChatId(null);
  }, []);

  // Update current chat ID and ensure chat exists
  const handleSetCurrentChatId = useCallback((chatId: string | null) => {
    if (chatId === null) {
      setCurrentChatId(null);
      return;
    }
    
    // Verify chat exists
    const chat = storage.getChat(chatId);
    if (chat) {
      setCurrentChatId(chatId);
    } else {
      console.error(`Chat ${chatId} not found`);
    }
  }, []);

  return {
    // Chat data
    chats: filteredChats,
    currentChat,
    currentChatId,
    
    // Search
    searchQuery,
    setSearchQuery,
    
    // Chat management
    setCurrentChatId: handleSetCurrentChatId,
    createNewChat,
    addMessage,
    deleteChat,
    renameChat,
    clearAllChats,
  };
}

