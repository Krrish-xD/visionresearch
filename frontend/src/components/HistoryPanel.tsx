import { useState, useEffect } from 'react';
import { Clock, CheckCircle2, AlertCircle } from 'lucide-react';
import styles from './HistoryPanel.module.css';

interface HistoryItem {
  id: string;
  filename: string;
  status: string;
  created_at: string;
  total_time_ms: number;
}

interface HistoryPanelProps {
  onClose: () => void;
  onSelectHistory: (id: string) => void;
}

export function HistoryPanel({ onClose, onSelectHistory }: HistoryPanelProps) {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/history')
      .then(r => r.json())
      .then(data => {
        setHistory(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load history", err);
        setLoading(false);
      });
  }, []);

  return (
    <div className={styles.overlay}>
      <div className={styles.panel}>
        <div className={styles.header}>
          <h2>Analysis History</h2>
          <button onClick={onClose} className={styles.closeBtn}>&times;</button>
        </div>
        
        <div className={styles.content}>
          {loading ? (
            <div className={styles.loading}>Loading history...</div>
          ) : history.length === 0 ? (
            <div className={styles.empty}>No past analyses found.</div>
          ) : (
            <div className={styles.list}>
              {history.map(item => (
                <div 
                  key={item.id} 
                  className={styles.historyCard}
                  onClick={() => onSelectHistory(item.id)}
                >
                  <div className={styles.cardInfo}>
                    <h4>{item.filename}</h4>
                    <div className={styles.meta}>
                      <Clock size={14} /> 
                      {new Date(item.created_at).toLocaleString()}
                    </div>
                  </div>
                  
                  <div className={styles.status}>
                    {item.status === 'complete' ? (
                      <span className={styles.complete}><CheckCircle2 size={16} /> {Math.round(item.total_time_ms / 1000)}s</span>
                    ) : item.status === 'error' ? (
                      <span className={styles.error}><AlertCircle size={16} /> Failed</span>
                    ) : (
                      <span className={styles.pending}>Processing...</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
