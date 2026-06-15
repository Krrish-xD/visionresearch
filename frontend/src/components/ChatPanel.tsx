import { useState, useEffect, useRef } from 'react';
import { Send, Bot, User } from 'lucide-react';
import styles from './ChatPanel.module.css';

interface ChatPanelProps {
  taskId: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const WELCOME: Message = { role: 'assistant', content: 'Ask me anything about this image!' };

export function ChatPanel({ taskId }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Load persisted chat history whenever the taskId changes
  useEffect(() => {
    if (!taskId) return;
    fetch(`/api/history/${taskId}`)
      .then(r => r.json())
      .then(data => {
        const history: Message[] = data.chat_history || [];
        setMessages(history.length > 0 ? history : [WELCOME]);
      })
      .catch(() => setMessages([WELCOME]));
  }, [taskId]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`/api/analyze/${taskId}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage })
      });

      if (!response.ok) throw new Error('Failed to fetch response');
      const data = await response.json();
      
      // Sync the full history from the server (source of truth)
      if (data.history && data.history.length > 0) {
        setMessages(data.history);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: data.answer || data.error }]);
      }
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.messageList}>
        {messages.map((msg, i) => (
          <div key={i} className={`${styles.message} ${styles[msg.role]}`}>
            <div className={styles.avatar}>
              {msg.role === 'assistant' ? <Bot size={16} /> : <User size={16} />}
            </div>
            <div className={styles.content}>{msg.content}</div>
          </div>
        ))}
        {isLoading && (
          <div className={`${styles.message} ${styles.assistant}`}>
            <div className={styles.avatar}><Bot size={16} /></div>
            <div className={styles.content}>
              <div className={styles.typingIndicator}>
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form className={styles.inputArea} onSubmit={handleSubmit}>
        <input 
          type="text" 
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask a question..."
          className={styles.input}
          disabled={isLoading}
        />
        <button type="submit" className={styles.sendBtn} disabled={isLoading || !input.trim()}>
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}
