import { useState, useCallback, useEffect, useMemo } from 'react';
import { useSSE } from './useSSE';
import { AnalysisResult, ModuleState } from '../types/analysis';

export function useAnalysis() {
  const { events, isConnected, lastEvent, connect, disconnect, error: sseError } = useSSE();
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [uploadedImageFile, setUploadedImageFile] = useState<File | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  
  const reset = useCallback(() => {
    disconnect();
    setIsAnalyzing(false);
    setUploadError(null);
    setTaskId(null);
    if (uploadedImageUrl) {
      URL.revokeObjectURL(uploadedImageUrl);
    }
    setUploadedImageUrl(null);
    setUploadedImageFile(null);
  }, [disconnect, uploadedImageUrl]);

  const uploadAndAnalyze = useCallback(async (file: File) => {
    reset();
    setIsAnalyzing(true);
    setUploadedImageFile(file);
    setUploadedImageUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      setTaskId(data.task_id);
      
      // Initialize SSE stream
      connect(data.task_id);
      
    } catch (err: any) {
      setUploadError(err.message || 'Failed to upload image');
      setIsAnalyzing(false);
    }
  }, [connect, reset]);
  
  // Derived state from events array guarantees we don't miss any events due to React batching
  const moduleStates = useMemo(() => {
    if (!taskId) return {};
    const states: Record<string, ModuleState> = {};
    events.forEach((evt: any) => {
      if (evt.event === 'pipeline_start' && evt.results?.modules) {
        evt.results.modules.forEach((mod: string) => {
          states[mod] = { name: mod, display_name: mod, status: 'pending' };
        });
      } else if (
        evt.event !== 'pipeline_start' &&
        evt.event !== 'pipeline_complete' &&
        evt.event !== 'pipeline_error' &&
        evt.module && evt.module !== 'pipeline'
      ) {
        const existing = states[evt.module] || {
          name: evt.module,
          display_name: evt.display_name || evt.module,
          status: 'pending',
        };
        states[evt.module] = {
          ...existing,
          display_name: evt.display_name || existing.display_name,
          status: evt.status,
          results: evt.results || existing.results,
          timing_ms: evt.timing_ms || existing.timing_ms,
          error: evt.error || existing.error,
        };
      }
    });
    return states;
  }, [events, taskId]);

  const analysisResult = useMemo(() => {
    if (!taskId) return {};
    let result: Partial<AnalysisResult> = {};
    events.forEach((evt: any) => {
      if (evt.event === 'pipeline_complete' && evt.results) {
        result = evt.results;
      } else if (evt.status === 'complete' && evt.results && evt.event !== 'pipeline_complete') {
        result = { ...result, ...evt.results };
      }
    });
    return result;
  }, [events, taskId]);

  // Handle stream termination
  useEffect(() => {
    if (events.length === 0) return;
    const latest: any = events[events.length - 1];
    if (latest.event === 'pipeline_complete') {
      setIsAnalyzing(false);
    } else if (latest.event === 'pipeline_error') {
      setIsAnalyzing(false);
      setUploadError(latest.error || 'Pipeline error');
    }
  }, [events]);

  // Progress calculation
  const totalModules = Object.keys(moduleStates).length;
  const completedModules = Object.values(moduleStates).filter(m => m.status === 'complete').length;
  const progressPercent = totalModules > 0 ? Math.round((completedModules / totalModules) * 100) : 0;

  return {
    uploadAndAnalyze,
    isAnalyzing,
    progress: progressPercent,
    error: uploadError || sseError,
    moduleStates,
    analysisResult,
    uploadedImageFile,
    uploadedImageUrl,
    reset,
    taskId
  };
}
