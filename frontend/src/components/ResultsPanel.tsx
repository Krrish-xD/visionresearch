import { useState } from 'react';
import { Box, Type, Info, ShieldAlert, Palette, ScanFace, ScanLine } from 'lucide-react';
import { ModuleState } from '../types/analysis';
import { ModuleCard } from './ModuleCard';
import styles from './ResultsPanel.module.css';

interface ResultsPanelProps {
  moduleStates: Record<string, ModuleState>;
  onHoverObject?: (id: string | null) => void;
}

type TabType = 'visual' | 'text' | 'meta';

export function ResultsPanel({ moduleStates, onHoverObject }: ResultsPanelProps) {
  const [activeTab, setActiveTab] = useState<TabType>('visual');

  // Group modules by category
  const groups = {
    visual: ['object_detection', 'faces', 'pose', 'depth', 'segmentation'],
    text: ['caption', 'ocr', 'tags'],
    meta: ['metadata', 'colors', 'nsfw', 'embedding', 'classification']
  };

  const renderModuleCard = (name: string, icon: React.ReactNode) => {
    const state = moduleStates[name];
    if (!state) return null;
    return <ModuleCard key={name} state={state} icon={icon} onHoverObject={onHoverObject} />;
  };

  return (
    <div className={styles.panelContainer}>
      
      {/* Tabs Header */}
      <div className={styles.tabsHeader}>
        <button 
          className={`${styles.tab} ${activeTab === 'visual' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('visual')}
        >
          <ScanLine size={16} /> Visual
        </button>
        <button 
          className={`${styles.tab} ${activeTab === 'text' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('text')}
        >
          <Type size={16} /> Text
        </button>
        <button 
          className={`${styles.tab} ${activeTab === 'meta' ? styles.activeTab : ''}`}
          onClick={() => setActiveTab('meta')}
        >
          <Info size={16} /> Meta
        </button>
      </div>

      {/* Scrollable Content Area */}
      <div className={styles.scrollArea}>
        {activeTab === 'visual' && (
          <div className="animate-fade-in">
            {renderModuleCard('object_detection', <Box size={18} />)}
            {renderModuleCard('faces', <ScanFace size={18} />)}
            {renderModuleCard('pose', <ScanLine size={18} />)}
            {renderModuleCard('depth', <ScanLine size={18} />)}
            {renderModuleCard('segmentation', <ScanLine size={18} />)}
          </div>
        )}

        {activeTab === 'text' && (
          <div className="animate-fade-in">
            {renderModuleCard('caption', <Type size={18} />)}
            {renderModuleCard('ocr', <Type size={18} />)}
            {renderModuleCard('tags', <Type size={18} />)}
          </div>
        )}

        {activeTab === 'meta' && (
          <div className="animate-fade-in">
            {renderModuleCard('metadata', <Info size={18} />)}
            {renderModuleCard('colors', <Palette size={18} />)}
            {renderModuleCard('nsfw', <ShieldAlert size={18} />)}
            {renderModuleCard('classification', <Info size={18} />)}
            {renderModuleCard('embedding', <Info size={18} />)}
          </div>
        )}
      </div>

    </div>
  );
}
