// In src/types.ts
export interface Message {
    text: string;
    isUser: boolean;
    type?: string;
    citations?: { page: number; text: string }[]; // New field for structured citations
  }
  
  export interface ChatWindowProps {
    messages: Message[];
    onAgentClick: (agent: string) => void;
    onSend: (text: string) => void;
    jumpToPage?: (page: number) => void;
  }