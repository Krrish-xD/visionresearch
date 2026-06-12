import { useEffect, useState } from 'react';
import { ModuleState } from '../types/analysis';
import styles from './ProgressBar.module.css';

interface ProgressBarProps {
  progress: number;
  isAnalyzing: boolean;
  moduleStates: Record<string, ModuleState>;
  error?: string | null;
}

export function ProgressBar({ progress, isAnalyzing, moduleStates, error }: ProgressBarProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let interval: ReturnType<typeof setTimeout>;
    if (isAnalyzing) {
      setElapsed(0);
      interval = setInterval(() => {
        setElapsed(prev => prev + 100);
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const formatTime = (ms: number) => {
    return (ms / 1000).toFixed(1) + 's';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete': return '✅';
      case 'running': return '🔄';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  // Group modules for the mini tracker
  const coreModules = ['metadata', 'colors', 'object_detection', 'nsfw', 'caption'];

  if (!isAnalyzing && progress === 0 && !error) return null;

  return (
    <div className={`glass-panel ${styles.container}`}>
      <div className={styles.topRow}>
        <div className={styles.statusText}>
          {error ? (
            <span className={styles.errorText}>Pipeline Error</span>
          ) : isAnalyzing ? (
            <span>Processing... {progress}%</span>
          ) : (
            <span className={styles.successText}>Analysis Complete</span>
          )}
        </div>
        <div className={styles.timeElapsed}>
          {formatTime(elapsed)} elapsed
        </div>
      </div>

      <div className={styles.barBackground}>
        <div 
          className={`${styles.barFill} ${isAnalyzing ? styles.barAnimated : ''} ${error ? styles.barError : ''}`}
          style={{ width: `${Math.max(progress, 2)}%` }}
        />
      </div>

      <div className={styles.moduleTracker}>
        {coreModules.map(name => {
          const state = moduleStates[name];
          if (!state) return null;
          return (
            <span key={name} className={styles.trackerItem} title={state.display_name}>
              {getStatusIcon(state.status)} <span className={styles.trackerName}>{state.display_name.split(' ')[0]}</span>
            </span>
          );
        })}
      </div>
      
      {error && (
        <div className={styles.errorMessage}>{error}</div>
      )}
    </div>
  );
}
