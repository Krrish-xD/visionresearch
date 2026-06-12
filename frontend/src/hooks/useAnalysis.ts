import { useState, useCallback, useEffect } from 'react';
import { useSSE } from './useSSE';
import { AnalysisResult, ModuleState } from '../types/analysis';

export function useAnalysis() {
  const { events, isConnected, lastEvent, connect, disconnect, error: sseError } = useSSE();
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [uploadedImageFile, setUploadedImageFile] = useState<File | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  
  // Track individual module states
  const [moduleStates, setModuleStates] = useState<Record<string, ModuleState>>({});
  
  // The final aggregated result
  const [analysisResult, setAnalysisResult] = useState<Partial<AnalysisResult>>({});

  const reset = useCallback(() => {
    disconnect();
    setIsAnalyzing(false);
    setUploadError(null);
    setTaskId(null);
    setModuleStates({});
    setAnalysisResult({});
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

  // Process incoming SSE events
  useEffect(() => {
    if (!lastEvent) return;

    if (lastEvent.module === 'pipeline_start' || lastEvent.module === 'pipeline_error' || lastEvent.module === 'pipeline_complete') {
      // Handle pipeline level events
      if (lastEvent.module === 'pipeline_complete') {
        setIsAnalyzing(false);
        if (lastEvent.results) {
           setAnalysisResult(lastEvent.results);
        }
      } else if (lastEvent.module === 'pipeline_error') {
        setIsAnalyzing(false);
        setUploadError(lastEvent.error || 'Pipeline error');
      } else if (lastEvent.module === 'pipeline_start') {
         // Initialize modules based on the start event
         if (lastEvent.results?.modules) {
           const initialStates: Record<string, ModuleState> = {};
           lastEvent.results.modules.forEach((mod: string) => {
             initialStates[mod] = { name: mod, display_name: mod, status: 'pending' };
           });
           setModuleStates(initialStates);
         }
      }
      return;
    }

    // Handle module level events
    setModuleStates(prev => {
      const existing = prev[lastEvent.module] || { name: lastEvent.module, display_name: lastEvent.display_name || lastEvent.module, status: 'pending' };
      
      return {
        ...prev,
        [lastEvent.module]: {
          ...existing,
          display_name: lastEvent.display_name || existing.display_name,
          status: lastEvent.status,
          results: lastEvent.results || existing.results,
          timing_ms: lastEvent.timing_ms || existing.timing_ms,
          error: lastEvent.error || existing.error
        }
      };
    });

    // Progressively merge results into analysisResult
    if (lastEvent.status === 'complete' && lastEvent.results) {
      setAnalysisResult(prev => ({
        ...prev,
        ...lastEvent.results
      }));
    }

  }, [lastEvent]);

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
