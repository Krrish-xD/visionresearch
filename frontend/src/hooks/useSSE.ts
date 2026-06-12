import { useState, useEffect, useRef, useCallback } from 'react';
import { ModuleEvent, ModuleStatus } from '../types/analysis';

interface UseSSEResult {
  events: ModuleEvent[];
  isConnected: boolean;
  error: string | null;
  lastEvent: ModuleEvent | null;
  connect: (taskId: string) => void;
  disconnect: () => void;
}

export function useSSE(): UseSSEResult {
  const [events, setEvents] = useState<ModuleEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastEvent, setLastEvent] = useState<ModuleEvent | null>(null);
  
  const eventSourceRef = useRef<EventSource | null>(null);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsConnected(false);
    }
  }, []);

  const connect = useCallback((taskId: string) => {
    disconnect();
    setEvents([]);
    setError(null);
    setIsConnected(false);

    try {
      const url = `/api/analyze/${taskId}/stream`;
      const source = new EventSource(url);
      eventSourceRef.current = source;

      source.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      source.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          
          // Map backend event structure to frontend ModuleEvent
          // Backend sends: { event: 'module_complete', module: 'yolo', display_name: '...', status: '...', results: {...} }
          const moduleEvent: ModuleEvent = {
            module: data.module || data.event,
            status: mapBackendEventToStatus(data.event, data.status),
            results: data.results,
            progress: data.progress,
            error: data.error,
            display_name: data.display_name,
            timing_ms: data.timing_ms
          };

          setLastEvent(moduleEvent);
          setEvents((prev) => [...prev, moduleEvent]);
          
          if (data.event === 'pipeline_complete' || data.event === 'pipeline_error') {
            source.close();
            setIsConnected(false);
          }
        } catch (err) {
          console.error("Failed to parse SSE event", err);
        }
      };

      source.onerror = () => {
        // SSE auto-reconnects, but if it's a hard error we might want to close
        setError("Connection lost. Attempting to reconnect...");
        setIsConnected(false);
      };
    } catch (err: any) {
      setError(err.message || "Failed to connect to stream");
      setIsConnected(false);
    }
  }, [disconnect]);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return { events, isConnected, error, lastEvent, connect, disconnect };
}

// Helper to map backend event strings to our UI status enum
function mapBackendEventToStatus(eventName: string, statusText?: string): ModuleStatus {
  if (eventName === 'pipeline_start') return 'running';
  if (eventName === 'pipeline_complete') return 'complete';
  if (eventName === 'pipeline_error') return 'error';
  
  if (eventName === 'module_start') return 'running';
  if (eventName === 'module_complete') return 'complete';
  if (eventName === 'module_error') return 'error';
  
  if (statusText === 'running') return 'running';
  if (statusText === 'complete') return 'complete';
  if (statusText === 'error') return 'error';
  
  return 'pending';
}
