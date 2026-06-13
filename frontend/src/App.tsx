import { useState } from 'react';
import { Microscope, Download, Copy, Settings, History, MessageSquare } from 'lucide-react';
import { ImageUpload } from './components/ImageUpload';
import { TaskRunner } from './components/TaskRunner';
import styles from './App.module.css';

import { HistoryPanel } from './components/HistoryPanel';

interface TaskSession {
  id: string;
  file?: File;
  historyId?: string;
}

function App() {
  const [sessions, setSessions] = useState<TaskSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  const handleUpload = (files: File[]) => {
    const newSessions = files.map(f => ({
      id: Math.random().toString(36).substring(7),
      file: f
    }));
    setSessions(prev => [...prev, ...newSessions]);
    if (!activeSessionId && newSessions.length > 0) {
      setActiveSessionId(newSessions[0].id);
    }
  };

  const handleSelectHistory = (id: string) => {
    setShowHistory(false);
    const existing = sessions.find(s => s.historyId === id);
    if (existing) {
      setActiveSessionId(existing.id);
      return;
    }
    const newId = Math.random().toString(36).substring(7);
    setSessions(prev => [...prev, { id: newId, historyId: id }]);
    setActiveSessionId(newId);
  };

  const handleCopyJSON = () => {
    alert("JSON copy will be implemented shortly.");
  };

  return (
    <div className={styles.appContainer}>
      <header className={`glass-panel ${styles.header}`}>
        <div className={styles.logo}>
          <Microscope size={28} className={styles.logoIcon} />
          <h1>VisionResearch</h1>
        </div>
        
        <div className={styles.headerActions}>
          <button className={styles.actionBtn} onClick={handleCopyJSON} disabled={sessions.length === 0}>
            <Copy size={18} /> <span className={styles.btnText}>Copy JSON</span>
          </button>
          <div className={styles.divider} />
          <button className={styles.iconBtn} title="History" onClick={() => setShowHistory(true)}>
            <History size={20} />
          </button>
          <button className={styles.iconBtn} title="Settings">
            <Settings size={20} />
          </button>
        </div>
      </header>

      {showHistory && (
        <HistoryPanel 
          onClose={() => setShowHistory(false)} 
          onSelectHistory={handleSelectHistory} 
        />
      )}

      <main className={styles.main}>
        {sessions.length === 0 ? (
          <div className={styles.uploadContainer}>
            <div className={styles.welcomeText}>
              <h2>Maximum Image Intelligence</h2>
              <p>Upload images to run a comprehensive analysis pipeline.</p>
            </div>
            <ImageUpload onUpload={handleUpload} />
          </div>
        ) : (
          <div className={styles.workspace}>
            <div className={styles.sidebar} id="sidebar-portal">
              {/* TaskRunner components will portal their thumbnails here */}
              <div className={styles.addMoreBtn}>
                 <ImageUpload onUpload={handleUpload} />
              </div>
            </div>
            
            <div className={styles.workspaceContent}>
              {sessions.map(session => (
                <TaskRunner 
                  key={session.id} 
                  file={session.file} 
                  historyId={session.historyId}
                  isActive={session.id === activeSessionId}
                  onSelect={() => setActiveSessionId(session.id)}
                />
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
