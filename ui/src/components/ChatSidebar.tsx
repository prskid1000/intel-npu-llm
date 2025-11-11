import { useState, useRef, useEffect } from 'react';
import { Plus, Search, Trash2, MessageSquare, Edit2, Check, X } from 'lucide-react';
import { useChatHistoryContext } from '../contexts/ChatHistoryContext';
import { Modal } from './Modal';
import { AvailableTools } from './AvailableTools';
import { MCPServerManager } from './MCPServerManager';
import { Tool } from '../services/apiService';

interface ChatSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  availableTools?: Tool[];
  toolsLoading?: boolean;
  toolsError?: string | null;
}

export function ChatSidebar({ 
  isOpen, 
  onToggle,
  availableTools = [],
  toolsLoading = false,
  toolsError = null
}: ChatSidebarProps) {
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [clearAllModalOpen, setClearAllModalOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const editInputRef = useRef<HTMLInputElement>(null);

  const {
    chats,
    currentChatId,
    searchQuery,
    setCurrentChatId,
    setSearchQuery,
    createNewChat,
    deleteChat,
    renameChat,
    clearAllChats,
  } = useChatHistoryContext();

  const handleNewChat = () => {
    const newChatId = createNewChat();
    if (newChatId) {
      setCurrentChatId(newChatId);
    }
    onToggle();
  };

  const handleChatSelect = (chatId: string) => {
    setCurrentChatId(chatId);
    onToggle();
  };

  const handleDeleteChat = (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    setChatToDelete(chatId);
    setDeleteModalOpen(true);
  };

  const confirmDelete = () => {
    if (chatToDelete) {
      deleteChat(chatToDelete);
      setChatToDelete(null);
    }
  };

  const confirmClearAll = () => {
    clearAllChats();
    setClearAllModalOpen(false);
  };

  const handleRenameStart = (e: React.MouseEvent, chatId: string, currentTitle: string) => {
    e.stopPropagation();
    setEditingChatId(chatId);
    setEditTitle(currentTitle);
  };

  const handleRenameSave = (chatId: string) => {
    if (editTitle.trim()) {
      renameChat(chatId, editTitle.trim());
    }
    setEditingChatId(null);
    setEditTitle('');
  };

  const handleRenameCancel = () => {
    setEditingChatId(null);
    setEditTitle('');
  };

  useEffect(() => {
    if (editingChatId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingChatId]);

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-50 w-64 bg-dark-surface border-r border-dark-border transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        <div className="flex flex-col h-full">
        {/* Header */}
        <div className="p-4 border-b border-dark-border">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center gap-2 px-4 py-2 bg-dark-accent hover:bg-dark-accentHover rounded-lg transition-colors"
          >
            <Plus className="w-5 h-5" />
            <span>New Chat</span>
          </button>
        </div>

        {/* Search */}
        <div className="p-4 border-b border-dark-border">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-dark-textSecondary" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chats..."
              className="w-full pl-10 pr-4 py-2 bg-dark-bg border border-dark-border rounded-lg focus:outline-none focus:ring-2 focus:ring-dark-accent text-sm"
            />
          </div>
          {chats.length > 0 && (
            <button
              onClick={() => setClearAllModalOpen(true)}
              className="mt-2 w-full flex items-center justify-center gap-2 px-3 py-1.5 text-xs text-dark-textSecondary hover:text-dark-error hover:bg-dark-surfaceHover rounded transition-colors"
            >
              <Trash2 className="w-3 h-3" />
              Clear All Chats
            </button>
          )}
        </div>

        {/* Chat list */}
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          {chats.length === 0 ? (
            <div className="p-4 text-center text-dark-textSecondary text-sm">
              <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No chats yet</p>
              <p className="text-xs mt-1">Start a new conversation</p>
            </div>
          ) : (
            <div className="p-2">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => !editingChatId && handleChatSelect(chat.id)}
                  className={`group relative p-3 mb-1 rounded-lg cursor-pointer transition-colors ${
                    currentChatId === chat.id
                      ? 'bg-dark-accent/20'
                      : 'hover:bg-dark-surfaceHover'
                  } ${editingChatId === chat.id ? 'bg-dark-surfaceHover' : ''}`}
                >
                  {editingChatId === chat.id ? (
                    <div className="flex items-center gap-2">
                      <input
                        ref={editInputRef}
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleRenameSave(chat.id);
                          } else if (e.key === 'Escape') {
                            handleRenameCancel();
                          }
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="flex-1 px-2 py-1 bg-dark-bg border border-dark-accent rounded text-sm focus:outline-none focus:ring-1 focus:ring-dark-accent"
                        maxLength={50}
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRenameSave(chat.id);
                        }}
                        className="p-1 hover:bg-dark-surface rounded transition-colors text-dark-accent"
                        aria-label="Save"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRenameCancel();
                        }}
                        className="p-1 hover:bg-dark-surface rounded transition-colors text-dark-textSecondary"
                        aria-label="Cancel"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">
                          {chat.title}
                        </p>
                        <p className="text-xs text-dark-textSecondary mt-1">
                          {new Date(chat.updatedAt).toLocaleDateString()}
                        </p>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => handleRenameStart(e, chat.id, chat.title)}
                          className="p-1 hover:bg-dark-surface rounded transition-colors text-dark-textSecondary hover:text-dark-accent"
                          aria-label="Rename chat"
                        >
                          <Edit2 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => handleDeleteChat(e, chat.id)}
                          className="p-1 hover:bg-dark-surface rounded transition-colors text-dark-textSecondary hover:text-dark-error"
                          aria-label="Delete chat"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Available Tools */}
        <AvailableTools 
          tools={availableTools} 
          isLoading={toolsLoading}
          error={toolsError}
          onRefresh={() => {
            // Trigger tools refresh via event
            if (window.dispatchEvent) {
              window.dispatchEvent(new CustomEvent('refreshTools'));
            }
          }}
        />

        {/* MCP Server Manager */}
        <MCPServerManager />

        {/* Footer */}
        <div className="p-4 border-t border-dark-border">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-dark-accent flex items-center justify-center text-white font-semibold text-sm">
              N
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">NPU Chat</p>
              <p className="text-xs text-dark-textSecondary">OpenVINO GenAI</p>
            </div>
          </div>
        </div>
        </div>
      </aside>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={deleteModalOpen}
        onClose={() => {
          setDeleteModalOpen(false);
          setChatToDelete(null);
        }}
        onConfirm={confirmDelete}
        title="Delete Chat"
        message="Are you sure you want to delete this chat? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        type="danger"
      />

      {/* Clear All Confirmation Modal */}
      <Modal
        isOpen={clearAllModalOpen}
        onClose={() => setClearAllModalOpen(false)}
        onConfirm={confirmClearAll}
        title="Clear All Chats"
        message={`Are you sure you want to delete all ${chats.length} chats? This action cannot be undone.`}
        confirmText="Clear All"
        cancelText="Cancel"
        type="danger"
      />
    </>
  );
}

