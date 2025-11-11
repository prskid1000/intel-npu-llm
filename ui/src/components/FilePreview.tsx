import { useEffect } from 'react';
import { X } from 'lucide-react';

interface FilePreviewProps {
  file: File;
  type: 'image' | 'video' | 'audio';
  onRemove: () => void;
}

export function FilePreview({ file, type, onRemove }: FilePreviewProps) {
  const url = URL.createObjectURL(file);

  useEffect(() => {
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [url]);

  return (
    <div className="relative inline-block mt-2">
      {type === 'image' && (
        <div className="relative group">
          <img
            src={url}
            alt={file.name}
            className="max-w-xs max-h-48 rounded-lg object-cover"
          />
          <button
            onClick={onRemove}
            className="absolute top-2 right-2 p-1 bg-dark-surface/80 hover:bg-dark-surface rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}

