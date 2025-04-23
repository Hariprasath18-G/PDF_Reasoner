import React, { useState } from 'react';
import styles from '../styles/Home.module.css';
import AgentButtons from './AgentButtons';

interface Message {
  text: string;
  isUser: boolean;
  type?: string;
  isLoading?: boolean;
}

interface ChatWindowProps {
  messages: Message[];
  onAgentClick: (agent: string) => void;
  onSend: (text: string) => void;
  jumpToPage?: (page: number) => void;
  isSending?: boolean;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ 
  messages, 
  onAgentClick, 
  onSend, 
  jumpToPage,
  isSending 
}) => {
  const [inputText, setInputText] = useState('');

  const handlePageLinkClick = (pageNumber: number) => {
    jumpToPage?.(pageNumber);
  };

  const renderMessageContent = (text: string) => {
    const parts = [];
    let lastIndex = 0;
    const pageRegex = /\[Page (\d+)\]/g;

    text.replace(pageRegex, (match, p1, offset) => {
      if (offset > lastIndex) {
        parts.push(<span key={parts.length}>{text.slice(lastIndex, offset)}</span>);
      }
      const pageNumber = parseInt(p1, 10);
      parts.push(
        <span
          key={parts.length}
          className={styles.pageLink}
          onClick={() => handlePageLinkClick(pageNumber)}
          style={{ color: '#1e90ff', cursor: 'pointer', textDecoration: 'underline' }}
        >
          {match}
        </span>
      );
      lastIndex = offset + match.length;
      return match;
    });

    if (lastIndex < text.length) {
      parts.push(<span key={parts.length}>{text.slice(lastIndex)}</span>);
    }

    return parts;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim()) {
      onSend(inputText);
      setInputText('');
    }
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatWindow}>
        {messages.map((message, index) => (
          <div
            key={index}
            className={`${styles.message} ${
              message.isUser ? styles.userMessage : styles.botMessage
            } ${message.type ? styles[message.type] : ''}`}
          >
            {message.isLoading ? (
              <div className={styles.typingIndicator}>
                <div className={styles.typingDot} />
                <div className={styles.typingDot} />
                <div className={styles.typingDot} />
              </div>
            ) : (
              renderMessageContent(message.text)
            )}
          </div>
        ))}
        {isSending && (
          <div className={`${styles.message} ${styles.botMessage}`}>
            <div className={styles.typingIndicator}>
              <div className={styles.typingDot} />
              <div className={styles.typingDot} />
              <div className={styles.typingDot} />
            </div>
          </div>
        )}
      </div>
      <div className={styles.agentButtons}>
        <AgentButtons onClick={onAgentClick} />
      </div>
      <div className={styles.textInputContainer}>
        <form onSubmit={handleSubmit} className={styles.textInput}>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your query..."
            className={styles.textInputField}
          />
          <button type="submit" disabled={isSending}>
            {isSending ? 'Sending...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatWindow;
