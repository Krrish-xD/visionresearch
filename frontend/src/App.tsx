import { useState } from 'react';
import { Microscope, Download, Copy, Settings } from 'lucide-react';
import { useAnalysis } from './hooks/useAnalysis';
import { ImageUpload } from './components/ImageUpload';
import { ImageCanvas } from './components/ImageCanvas';
import { ResultsPanel } from './components/ResultsPanel';
import { ProgressBar } from './components/ProgressBar';
import styles from './App.module.css';

function App() {
  const { 
    uploadAndAnalyze, 
    isAnalyzing, 
    progress, 
    error, 
    moduleStates, 
    analysisResult,
    uploadedImageUrl 
  } = useAnalysis();

  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  
  // Which overlays are visible on the canvas
  const [activeOverlays, setActiveOverlays] = useState({
    objects: true,
    faces: true,
    pose: true,
    depth: false,
    segmentation: false
  });

  const handleCopyJSON = () => {
    navigator.clipboard.writeText(JSON.stringify(analysisResult, null, 2));
    alert("JSON copied to clipboard!");
  };

  const handleExportJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(analysisResult, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `vision_analysis_${analysisResult.image_id || 'result'}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  return (
    <div className={styles.appContainer}>
      {/* Header */}
      <header className={`glass-panel ${styles.header}`}>
        <div className={styles.logo}>
          <Microscope size={28} className={styles.logoIcon} />
          <h1>VisionResearch</h1>
        </div>
        
        <div className={styles.headerActions}>
          <button className={styles.actionBtn} onClick={handleCopyJSON} disabled={!analysisResult.image_id}>
            <Copy size={18} /> <span className={styles.btnText}>Copy JSON</span>
          </button>
          <button className={styles.actionBtn} onClick={handleExportJSON} disabled={!analysisResult.image_id}>
            <Download size={18} /> <span className={styles.btnText}>Export</span>
          </button>
          <div className={styles.divider} />
          <button className={styles.iconBtn} title="Settings">
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className={styles.main}>
        {!uploadedImageUrl ? (
          /* Initial State: Upload Zone centered */
          <div className={styles.uploadContainer}>
            <div className={styles.welcomeText}>
              <h2>Maximum Image Intelligence</h2>
              <p>Upload an image to run a comprehensive analysis pipeline.</p>
            </div>
            <ImageUpload onUpload={uploadAndAnalyze} disabled={isAnalyzing} />
          </div>
        ) : (
          /* Analysis State: Split View */
          <div className={styles.splitView}>
            
            {/* Left Column: Canvas & Controls */}
            <div className={styles.leftCol}>
              <div className={styles.canvasWrapper}>
                <ImageCanvas 
                  imageUrl={uploadedImageUrl} 
                  analysisResult={analysisResult} 
                  hoveredObjectId={hoveredObjectId}
                  activeOverlays={activeOverlays}
                  onOverlayChange={(key, val) => setActiveOverlays(p => ({...p, [key]: val}))}
                />
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
            
            {/* Right Column: Results Panel */}
            <div className={styles.rightCol}>
              <ResultsPanel 
                moduleStates={moduleStates} 
                onHoverObject={setHoveredObjectId}
              />
            </div>
            
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
