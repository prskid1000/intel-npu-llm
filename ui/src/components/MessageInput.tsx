import { useState, useRef, ChangeEvent, FormEvent } from 'react';
import { Send, Paperclip, Mic, X, Loader2, Volume2, VolumeX, Wrench } from 'lucide-react';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';
import { FilePreview } from './FilePreview';

interface MessageInputProps {
  onSend: (
    text: string,
    audioFile?: File,
    imageFile?: File,
    videoFile?: File
  ) => void;
  isLoading?: boolean;
  disabled?: boolean;
  audioOutputEnabled?: boolean;
  onAudioOutputToggle?: () => void;
  toolCallingEnabled?: boolean;
  onToolCallingToggle?: () => void;
}

export function MessageInput({ 
  onSend, 
  isLoading = false, 
  disabled = false, 
  audioOutputEnabled = true,
  onAudioOutputToggle,
  toolCallingEnabled = true,
  onToolCallingToggle
}: MessageInputProps) {
  const [text, setText] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const {
    isRecording,
    audioBlob,
    error: recordingError,
    startRecording,
    stopRecording,
    clearRecording,
    getAudioFile,
  } = useVoiceRecorder();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    
    if (!text.trim() && !imageFile && !videoFile && !audioBlob) {
      return;
    }

    if (disabled || isLoading) return;

    const audioFile = getAudioFile();
    
    onSend(
      text.trim(),
      audioFile || undefined,
      imageFile || undefined,
      videoFile || undefined
    );

    // Reset form
    setText('');
    setImageFile(null);
    setVideoFile(null);
    clearRecording();
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const type = file.type.split('/')[0];
    
    if (type === 'image') {
      setImageFile(file);
    } else if (type === 'video') {
      setVideoFile(file);
    }

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  // Auto-resize textarea
  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  };

  return (
    <div className="border-t border-dark-border bg-dark-surface">
      <div className="max-w-4xl mx-auto p-4">
        {/* File previews */}
        {(imageFile || videoFile || audioBlob) && (
          <div className="mb-3 flex flex-wrap gap-2">
            {imageFile && (
              <FilePreview
                file={imageFile}
                type="image"
                onRemove={() => setImageFile(null)}
              />
            )}
            {videoFile && (
              <FilePreview
                file={videoFile}
                type="video"
                onRemove={() => setVideoFile(null)}
              />
            )}
            {audioBlob && (
              <div className="flex items-center gap-2 p-2 bg-dark-surface rounded-lg">
                <span className="text-sm text-dark-textSecondary">
                  Audio recorded
                </span>
                <button
                  onClick={clearRecording}
                  className="p-1 hover:bg-dark-surfaceHover rounded-full transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        )}

        {/* Recording error */}
        {recordingError && (
          <div className="mb-3 p-2 bg-dark-error/20 text-dark-error rounded text-sm">
            {recordingError}
          </div>
        )}

        {/* Input form */}
        <form onSubmit={handleSubmit} className="flex items-end gap-2">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={text}
              onChange={handleTextChange}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Shift+Enter for new line)"
              disabled={disabled || isLoading}
              rows={1}
              className="w-full px-4 py-3 pr-12 bg-dark-bg border border-dark-border rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-dark-accent focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed scrollbar-thin"
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
          </div>

          <div className="flex items-center gap-2">
            {/* File attachment button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || isLoading}
              className="p-3 rounded-full bg-dark-surface hover:bg-dark-surfaceHover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Attach file"
            >
              <Paperclip className="w-5 h-5" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            {/* Voice recording button */}
            <button
              type="button"
              onClick={toggleRecording}
              disabled={disabled || isLoading}
              className={`p-3 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                isRecording
                  ? 'bg-dark-error hover:bg-red-600 animate-pulse'
                  : 'bg-dark-surface hover:bg-dark-surfaceHover'
              }`}
              aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            >
              <Mic className="w-5 h-5" />
            </button>

            {/* Audio output toggle */}
            {onAudioOutputToggle && (
              <button
                type="button"
                onClick={onAudioOutputToggle}
                disabled={disabled || isLoading}
                className={`p-3 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  audioOutputEnabled
                    ? 'bg-dark-accent/20 hover:bg-dark-accent/30 text-dark-accent'
                    : 'bg-dark-surface hover:bg-dark-surfaceHover text-dark-textSecondary'
                }`}
                aria-label={audioOutputEnabled ? 'Disable audio output' : 'Enable audio output'}
                title={audioOutputEnabled ? 'Audio output enabled' : 'Audio output disabled'}
              >
                {audioOutputEnabled ? (
                  <Volume2 className="w-5 h-5" />
                ) : (
                  <VolumeX className="w-5 h-5" />
                )}
              </button>
            )}

            {/* Tool calling toggle */}
            {onToolCallingToggle && (
              <button
                type="button"
                onClick={onToolCallingToggle}
                disabled={disabled || isLoading}
                className={`p-3 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  toolCallingEnabled
                    ? 'bg-dark-accent/20 hover:bg-dark-accent/30 text-dark-accent'
                    : 'bg-dark-surface hover:bg-dark-surfaceHover text-dark-textSecondary'
                }`}
                aria-label={toolCallingEnabled ? 'Disable tool calling' : 'Enable tool calling'}
                title={toolCallingEnabled ? 'Tool calling enabled' : 'Tool calling disabled'}
              >
                <Wrench className="w-5 h-5" />
              </button>
            )}

            {/* Send button */}
            <button
              type="submit"
              disabled={disabled || isLoading || (!text.trim() && !imageFile && !videoFile && !audioBlob)}
              className="p-3 rounded-full bg-dark-accent hover:bg-dark-accentHover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Send message"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

