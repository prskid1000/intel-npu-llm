import { ToolCall } from '../services/apiService';
import { Wrench, CheckCircle2, XCircle } from 'lucide-react';

interface ToolCallDisplayProps {
  toolCalls: ToolCall[];
  result?: string;
  error?: string;
}

export function ToolCallDisplay({ toolCalls, result, error }: ToolCallDisplayProps) {
  return (
    <div className="mt-2 space-y-2">
      {toolCalls.map((toolCall) => {
        let parsedArgs: any = {};
        try {
          parsedArgs = JSON.parse(toolCall.function.arguments);
        } catch {
          parsedArgs = { raw: toolCall.function.arguments };
        }

        return (
          <div
            key={toolCall.id}
            className="bg-dark-surfaceSecondary rounded-lg p-3 border border-dark-border"
          >
            <div className="flex items-center gap-2 mb-2">
              <Wrench className="w-4 h-4 text-dark-accent" />
              <span className="font-semibold text-sm text-dark-text">
                {toolCall.function.name}
              </span>
            </div>
            
            <div className="text-xs text-dark-textSecondary mb-2">
              <div className="font-mono bg-dark-surface p-2 rounded">
                {JSON.stringify(parsedArgs, null, 2)}
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 text-red-400 text-xs">
                <XCircle className="w-3 h-3" />
                <span>{error}</span>
              </div>
            )}

            {result && !error && (
              <div className="flex items-center gap-2 text-green-400 text-xs">
                <CheckCircle2 className="w-3 h-3" />
                <span className="font-mono text-xs break-all">{result}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

