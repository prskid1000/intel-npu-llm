import { Tool } from '../services/apiService';
import { Wrench, ChevronDown, ChevronUp, Loader2, AlertCircle, RefreshCw } from 'lucide-react';
import { useState } from 'react';

interface AvailableToolsProps {
  tools: Tool[];
  isOpen?: boolean;
  onToggle?: () => void;
  isLoading?: boolean;
  error?: string | null;
  onRefresh?: () => void;
}

export function AvailableTools({ 
  tools, 
  isOpen: controlledOpen, 
  onToggle,
  isLoading = false,
  error = null,
  onRefresh
}: AvailableToolsProps) {
  const [internalOpen, setInternalOpen] = useState(true);
  const isOpen = controlledOpen !== undefined ? controlledOpen : internalOpen;
  const toggle = onToggle || (() => setInternalOpen(!internalOpen));

  return (
    <div className="border-t border-dark-border">
      <div className="flex items-center">
        <button
          onClick={toggle}
          className="flex-1 flex items-center justify-between p-4 hover:bg-dark-surfaceHover transition-colors"
        >
          <div className="flex items-center gap-2">
            <Wrench className="w-4 h-4 text-dark-accent" />
            <span className="text-sm font-medium text-dark-text">
              Available Tools {tools.length > 0 && `(${tools.length})`}
            </span>
          </div>
          {isOpen ? (
            <ChevronUp className="w-4 h-4 text-dark-textSecondary" />
          ) : (
            <ChevronDown className="w-4 h-4 text-dark-textSecondary" />
          )}
        </button>
        {onRefresh && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRefresh();
            }}
            className="p-2 hover:bg-dark-surfaceHover transition-colors text-dark-textSecondary hover:text-dark-accent"
            title="Refresh tools"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        )}
      </div>

      {isOpen && (
        <div className="px-4 pb-4 space-y-2 max-h-96 overflow-y-auto">
          {isLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-4 h-4 animate-spin text-dark-accent mr-2" />
              <span className="text-xs text-dark-textSecondary">Loading tools...</span>
            </div>
          ) : error ? (
            <div className="flex items-center gap-2 p-3 bg-dark-error/20 border border-dark-error/50 rounded-lg">
              <AlertCircle className="w-4 h-4 text-dark-error flex-shrink-0" />
              <div className="flex-1">
                <div className="text-xs font-medium text-dark-error mb-1">Error loading tools</div>
                <div className="text-xs text-dark-textSecondary">{error}</div>
              </div>
            </div>
          ) : tools.length === 0 ? (
            <div className="text-center py-4">
              <div className="text-xs text-dark-textSecondary">No tools available</div>
              <div className="text-xs text-dark-textSecondary mt-1">
                Make sure the server is running
              </div>
            </div>
          ) : (
            tools.map((tool, index) => (
              <div
                key={index}
                className="bg-dark-surfaceSecondary rounded-lg p-3 border border-dark-border"
              >
                <div className="font-semibold text-sm text-dark-text mb-1">
                  {tool.function.name}
                </div>
                <div className="text-xs text-dark-textSecondary mb-2">
                  {tool.function.description}
                </div>
                {tool.function.parameters && (
                  <details className="text-xs">
                    <summary className="cursor-pointer text-dark-textSecondary hover:text-dark-text">
                      Parameters
                    </summary>
                    <pre className="mt-2 p-2 bg-dark-surface rounded text-xs overflow-x-auto">
                      {JSON.stringify(tool.function.parameters, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

