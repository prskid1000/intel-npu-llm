import { createContext, useContext, ReactNode } from 'react';
import { useChatHistory } from '../hooks/useChatHistory';

interface ChatHistoryContextType {
  chats: ReturnType<typeof useChatHistory>['chats'];
  currentChat: ReturnType<typeof useChatHistory>['currentChat'];
  currentChatId: ReturnType<typeof useChatHistory>['currentChatId'];
  searchQuery: ReturnType<typeof useChatHistory>['searchQuery'];
  setSearchQuery: ReturnType<typeof useChatHistory>['setSearchQuery'];
  setCurrentChatId: ReturnType<typeof useChatHistory>['setCurrentChatId'];
  createNewChat: ReturnType<typeof useChatHistory>['createNewChat'];
  addMessage: ReturnType<typeof useChatHistory>['addMessage'];
  deleteChat: ReturnType<typeof useChatHistory>['deleteChat'];
  renameChat: ReturnType<typeof useChatHistory>['renameChat'];
  clearAllChats: ReturnType<typeof useChatHistory>['clearAllChats'];
}

const ChatHistoryContext = createContext<ChatHistoryContextType | undefined>(undefined);

export function ChatHistoryProvider({ children }: { children: ReactNode }) {
  const chatHistory = useChatHistory();

  return (
    <ChatHistoryContext.Provider value={chatHistory}>
      {children}
    </ChatHistoryContext.Provider>
  );
}

export function useChatHistoryContext() {
  const context = useContext(ChatHistoryContext);
  if (context === undefined) {
    throw new Error('useChatHistoryContext must be used within a ChatHistoryProvider');
  }
  return context;
}

