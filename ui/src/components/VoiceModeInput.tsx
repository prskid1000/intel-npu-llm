import { Mic, Square, X, Loader2, Image as ImageIcon } from 'lucide-react';
import { useState, useRef, ChangeEvent } from 'react';
import { FilePreview } from './FilePreview';

interface VoiceModeInputProps {
  isRecording: boolean;
  isLoading: boolean;
  disabled: boolean;
  recordingError: string | null;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onClearRecording: () => void;
  onSendWithMedia?: (imageFile?: File, videoFile?: File) => void;
}

export function VoiceModeInput({
  isRecording,
  isLoading,
  disabled,
  recordingError,
  onStartRecording,
  onStopRecording,
  onClearRecording,
  onSendWithMedia,
}: VoiceModeInputProps) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const type = file.type.split('/')[0];
    
    if (type === 'image') {
      setImageFile(file);
    } else if (type === 'video') {
      setVideoFile(file);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSendWithMedia = () => {
    if (onSendWithMedia) {
      onSendWithMedia(imageFile || undefined, videoFile || undefined);
      setImageFile(null);
      setVideoFile(null);
    }
  };

  return (
    <div className="border-t border-dark-border bg-dark-surface">
      <div className="max-w-4xl mx-auto p-4">
        {/* Recording status */}
        {isRecording && (
          <div className="mb-4 p-4 bg-dark-error/20 border border-dark-error/50 rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-dark-error rounded-full animate-pulse" />
              <span className="text-sm font-medium text-dark-error">
                Recording... Click stop when finished
              </span>
            </div>
          </div>
        )}

        {/* Loading status */}
        {isLoading && (
          <div className="mb-4 p-4 bg-dark-accent/20 border border-dark-accent/50 rounded-lg">
            <div className="flex items-center gap-3">
              <Loader2 className="w-4 h-4 animate-spin text-dark-accent" />
              <span className="text-sm font-medium text-dark-accent">
                Processing your voice...
              </span>
            </div>
          </div>
        )}

        {/* Recording error */}
        {recordingError && (
          <div className="mb-4 p-3 bg-dark-error/20 text-dark-error rounded text-sm">
            {recordingError}
          </div>
        )}

        {/* Media previews */}
        {(imageFile || videoFile) && (
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
          </div>
        )}

        {/* Main voice input area */}
        <div className="flex items-center gap-3">
          {/* Media attachment (optional) */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || isLoading || isRecording}
            className="p-3 rounded-full bg-dark-surface hover:bg-dark-surfaceHover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Attach media"
            title="Attach image or video (optional)"
          >
            <ImageIcon className="w-5 h-5" />
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            onChange={handleFileSelect}
            className="hidden"
          />

          {/* Main voice button */}
          <div className="flex-1 flex justify-center">
            <button
              type="button"
              onClick={isRecording ? onStopRecording : onStartRecording}
              disabled={disabled || isLoading}
              className={`relative p-6 rounded-full transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
                isRecording
                  ? 'bg-dark-error hover:bg-red-600 scale-110'
                  : 'bg-dark-accent hover:bg-dark-accentHover'
              }`}
              aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            >
              {isRecording ? (
                <Square className="w-8 h-8 text-white" />
              ) : (
                <Mic className="w-8 h-8 text-white" />
              )}
              {isRecording && (
                <div className="absolute inset-0 rounded-full bg-dark-error animate-ping opacity-75" />
              )}
            </button>
          </div>

          {/* Send with media button (only shown when media is attached) */}
          {onSendWithMedia && (imageFile || videoFile) && (
            <button
              type="button"
              onClick={handleSendWithMedia}
              disabled={disabled || isLoading || isRecording}
              className="px-4 py-2 rounded-lg bg-dark-accent hover:bg-dark-accentHover transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
            >
              Send with Media
            </button>
          )}

          {/* Clear button (only shown when recording) */}
          {isRecording && (
            <button
              type="button"
              onClick={onClearRecording}
              disabled={disabled || isLoading}
              className="p-3 rounded-full bg-dark-surface hover:bg-dark-surfaceHover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Cancel recording"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-4 text-center">
          <p className="text-xs text-dark-textSecondary">
            {isRecording
              ? 'Speak now. Click stop when finished.'
              : 'Click the microphone to start a voice conversation'}
          </p>
        </div>
      </div>
    </div>
  );
}

