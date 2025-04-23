import React from 'react';
import styles from '../styles/Home.module.css';

const AgentButtons: React.FC<{ onClick: (agent: string) => void }> = ({ onClick }) => {
  const agents = [
    { name: 'Summarize', icon: '📝' },
    { name: 'Abstract', icon: '🔍' },
    { name: 'Key Findings', icon: '🔑' },
    { name: 'Challenges', icon: '⚠️' }
  ];

  return (
    <div className={styles.agentButtons}>
      {agents.map((agent) => (
        <button
          key={agent.name}
          onClick={() => onClick(agent.name)}
          className={styles.agentButton}
          style={{
            background:
              agent.name === 'Summarize' ? 'linear-gradient(135deg, #4a90e2, #50e3c2)' :
              agent.name === 'Abstract' ? 'linear-gradient(135deg, #ff6b6b, #ffa3a3)' :
              agent.name === 'Key Findings' ? 'linear-gradient(135deg, #81c784, #a5d6a7)' :
              'linear-gradient(135deg, #ffb74d, #ffcc80)',
          }}
        >
          <span className={styles.agentIcon}>{agent.icon}</span>
          {agent.name}
        </button>
      ))}
    </div>
  );
};

export default AgentButtons;