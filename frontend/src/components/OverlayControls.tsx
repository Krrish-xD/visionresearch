import { Layers } from 'lucide-react';
import styles from './OverlayControls.module.css';

interface OverlayControlsProps {
  activeOverlays: Record<string, boolean>;
  onChange: (key: string, value: boolean) => void;
}

export function OverlayControls({ activeOverlays, onChange }: OverlayControlsProps) {
  return (
    <div className={`glass-panel ${styles.container}`}>
      <div className={styles.header}>
        <Layers size={14} />
        <span>Overlays</span>
      </div>
      <div className={styles.toggles}>
        {Object.entries(activeOverlays).map(([key, value]) => (
          <label key={key} className={styles.toggle}>
            <input 
              type="checkbox" 
              checked={value} 
              onChange={(e) => onChange(key, e.target.checked)} 
            />
            <span className={styles.label}>{key.charAt(0).toUpperCase() + key.slice(1)}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
