import React, { useState } from 'react';
import { ChevronDown, ChevronUp, CheckCircle, Loader2, XCircle, Clock, Info } from 'lucide-react';
import { ModuleState, DetectedObject, ColorInfo } from '../types/analysis';
import styles from './ModuleCard.module.css';

interface ModuleCardProps {
  state: ModuleState;
  icon?: React.ReactNode;
  onHoverObject?: (id: string | null) => void;
}

export function ModuleCard({ state, icon, onHoverObject }: ModuleCardProps) {
  const [expanded, setExpanded] = useState(state.status === 'complete');
  
  // Auto-expand when complete
  React.useEffect(() => {
    if (state.status === 'complete') {
      setExpanded(true);
    }
  }, [state.status]);

  const toggleExpand = () => setExpanded(!expanded);

  const renderStatusIcon = () => {
    switch (state.status) {
      case 'complete': return <CheckCircle size={18} className={styles.statusSuccess} />;
      case 'running': return <Loader2 size={18} className={`${styles.statusRunning} animate-pulse`} />;
      case 'error': return <XCircle size={18} className={styles.statusError} />;
      default: return <Clock size={18} className={styles.statusPending} />;
    }
  };

  const renderContent = () => {
    if (!state.results) return null;

    // Render based on module name
    switch (state.name) {
      case 'object_detection':
        const objects: DetectedObject[] = state.results.objects || [];
        if (objects.length === 0) return <div className={styles.emptyText}>No objects detected</div>;
        
        return (
          <div className={styles.listContainer}>
            {objects.map((obj, i) => (
              <div 
                key={i} 
                className={styles.listItem}
                onMouseEnter={() => onHoverObject?.(`obj-${i}`)}
                onMouseLeave={() => onHoverObject?.(null)}
              >
                <div className={styles.listItemHeader}>
                  <span className={styles.objectLabel}>{obj.label}</span>
                  <span className={styles.objectScore}>{(obj.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className={styles.progressBarBg}>
                  <div 
                    className={styles.progressBarFill} 
                    style={{ width: `${obj.confidence * 100}%`, backgroundColor: 'var(--color-module-objects)' }}
                  />
                </div>
              </div>
            ))}
          </div>
        );

      case 'caption':
        return (
          <div className={styles.captionContainer}>
            <p className={styles.captionBrief}>{state.results.caption}</p>
            {state.results.detailed_description && (
              <p className={styles.captionDetailed}>{state.results.detailed_description}</p>
            )}
          </div>
        );

      case 'colors':
        const colors: ColorInfo[] = state.results.colors || [];
        if (colors.length === 0) return null;
        
        return (
          <div className={styles.colorGrid}>
            {colors.map((c, i) => (
              <div key={i} className={styles.colorItem}>
                <div 
                  className={styles.colorSwatch} 
                  style={{ backgroundColor: c.hex }} 
                  title={c.name}
                />
                <div className={styles.colorDetails}>
                  <span className={styles.colorHex}>{c.hex}</span>
                  <span className={styles.colorPct}>{(c.percentage * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        );

      case 'metadata':
        const meta = state.results.metadata;
        if (!meta) return null;
        return (
          <table className={styles.metaTable}>
            <tbody>
              <tr><td>Dimensions</td><td>{meta.width} × {meta.height}</td></tr>
              <tr><td>Format</td><td>{meta.format} ({meta.mode})</td></tr>
              {meta.camera && <tr><td>Camera</td><td>{meta.camera}</td></tr>}
            </tbody>
          </table>
        );
        
      case 'nsfw':
        const nsfw = state.results.nsfw;
        if (!nsfw) return null;
        const isSafe = !nsfw.is_nsfw;
        return (
          <div className={`${styles.nsfwBadge} ${isSafe ? styles.nsfwSafe : styles.nsfwDanger}`}>
            <span className={styles.nsfwLabel}>{isSafe ? 'Safe Content' : 'NSFW Warning'}</span>
            <span className={styles.nsfwConfidence}>{(nsfw.confidence * 100).toFixed(1)}%</span>
          </div>
        );

      case 'ocr':
        const textRegions = state.results.text_regions || [];
        if (textRegions.length === 0) return <div className={styles.emptyText}>No text detected</div>;
        return (
          <div className={styles.textList}>
            {textRegions.map((tr: any, i: number) => (
              <div key={i} className={styles.textItem}>
                <span className={styles.textContent}>"{tr.content}"</span>
                <span className={styles.textConf}>{(tr.confidence * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        );

      case 'faces':
        const faces = state.results.faces || [];
        if (faces.length === 0) return <div className={styles.emptyText}>No faces detected</div>;
        return (
          <div className={styles.faceList}>
            {faces.map((f: any, i: number) => (
              <div key={i} className={styles.faceItem}>
                <div className={styles.faceHeader}>Face {i + 1}</div>
                <div className={styles.faceDetails}>
                  {f.age && <span>Age: {f.age}</span>}
                  {f.gender && <span>Gender: {f.gender}</span>}
                  {f.emotion && <span>Emotion: {f.emotion} ({(f.emotion_confidence * 100).toFixed(0)}%)</span>}
                </div>
              </div>
            ))}
          </div>
        );

      case 'pose':
        const poses = state.results.poses || [];
        if (poses.length === 0) return <div className={styles.emptyText}>No people detected</div>;
        return <div className={styles.simpleText}>{poses.length} person(s) detected with pose estimation.</div>;

      case 'depth':
      case 'segmentation':
        return <div className={styles.simpleText}>Map generated. Use the overlay toggle to view.</div>;

      case 'siglip':
        const tags = state.results.tags || [];
        return (
          <div className={styles.siglipContainer}>
            <div className={styles.tagsContainer}>
              {tags.length > 0 ? tags.map((t: string) => (
                <span key={t} className={styles.tagBadge}>{t}</span>
              )) : <span className={styles.emptyText}>No tags matched</span>}
            </div>
            {state.results.embedding && (
              <div className={styles.embeddingInfo}>
                <Info size={12} />
                <span>Generated {state.results.embedding.length}-d embedding</span>
              </div>
            )}
          </div>
        );

      default:
        // Generic JSON fallback
        return <pre className={styles.jsonBlock}>{JSON.stringify(state.results, null, 2)}</pre>;
    }
  };

  const getModuleColorClass = () => {
    // We map the module name to a CSS variable defined in index.css
    return styles[`module_${state.name}`] || styles.module_default;
  };

  return (
    <div className={`glass-panel ${styles.card} ${getModuleColorClass()}`}>
      <div className={styles.header} onClick={toggleExpand}>
        <div className={styles.headerLeft}>
          <div className={styles.iconWrapper}>{icon}</div>
          <span className={styles.title}>{state.display_name}</span>
        </div>
        <div className={styles.headerRight}>
          {state.timing_ms && <span className={styles.timing}>{state.timing_ms}ms</span>}
          {renderStatusIcon()}
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </div>
      
      {expanded && (
        <div className={`${styles.body} animate-slide-up`}>
          {state.error ? (
            <div className={styles.errorMessage}>{state.error}</div>
          ) : (
            renderContent()
          )}
        </div>
      )}
    </div>
  );
}
