import { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useAnalysis } from '../hooks/useAnalysis';
import { ImageCanvas } from './ImageCanvas';
import { ResultsPanel } from './ResultsPanel';
import { ProgressBar } from './ProgressBar';
import styles from '../App.module.css';

export function TaskRunner({ 
  file, 
  historyId,
  isActive, 
  onSelect 
}: { 
  file?: File; 
  historyId?: string;
  isActive: boolean; 
  onSelect: () => void;
}) {
  const { 
    uploadAndAnalyze, 
    loadHistory,
    isAnalyzing, 
    progress, 
    error, 
    moduleStates, 
    analysisResult,
    uploadedImageUrl,
    taskId 
  } = useAnalysis();

  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [activeOverlays, setActiveOverlays] = useState({
    objects: true,
    faces: true,
    pose: true,
    depth: false,
    segmentation: false
  });

  // Start analysis automatically when component mounts
  useEffect(() => {
    if (file) {
      uploadAndAnalyze(file);
    } else if (historyId) {
      loadHistory(historyId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  const sidebarNode = document.getElementById('sidebar-portal');

  return (
    <>
      {sidebarNode && createPortal(
        <div 
          className={`${styles.sidebarItem} ${isActive ? styles.activeSidebarItem : ''}`}
          onClick={onSelect}
        >
          {uploadedImageUrl && <img src={uploadedImageUrl} alt="thumb" className={styles.thumbnail} />}
          <div className={styles.thumbnailProgressBg}>
             <div 
               className={`${styles.thumbnailProgressFill} ${error ? styles.thumbnailError : ''}`} 
               style={{ width: `${progress}%` }} 
             />
          </div>
        </div>,
        sidebarNode
      )}
      
      <div style={{ display: isActive ? 'flex' : 'none', flex: 1, height: '100%', overflow: 'hidden' }}>
        <div className={styles.splitView}>
          <div className={styles.leftCol}>
            <div className={styles.canvasWrapper}>
              {uploadedImageUrl && (
                <ImageCanvas 
                  imageUrl={uploadedImageUrl} 
                  analysisResult={analysisResult} 
                  hoveredObjectId={hoveredObjectId}
                  activeOverlays={activeOverlays}
                  onOverlayChange={(key, val) => setActiveOverlays(p => ({...p, [key]: val}))}
                />
              )}
            </div>
            
            <div className={styles.progressWrapper}>
              <ProgressBar 
                progress={progress} 
                isAnalyzing={isAnalyzing} 
                moduleStates={moduleStates} 
                error={error} 
              />
            </div>
          </div>
          
          <div className={styles.rightCol}>
            <ResultsPanel 
              taskId={taskId}
              moduleStates={moduleStates} 
              onHoverObject={setHoveredObjectId}
            />
          </div>
        </div>
      </div>
    </>
  );
}
