import { useState, useEffect, useCallback, useRef } from 'react';
import Layout from '../components/Layout';
import dynamic from 'next/dynamic';
import { Message } from '../types';
import styles from '../styles/Home.module.css';
import axios from 'axios';

const ChatWindow = dynamic(
  () => import('../components/ChatWindow').then((mod) => mod.default),
  { ssr: false, loading: () => <div>Loading chat...</div> }
);

const PDFPreview = dynamic(
  () => import('../components/PDFPreview').then((mod) => mod.default),
  { ssr: false, loading: () => <div>Loading PDF viewer...</div> }
);

export default function Home() {
  const [pdfs, setPdfs] = useState<string[]>([]);
  const [selectedPDF, setSelectedPDF] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{message: string, type: 'uploading' | 'success'} | null>(null);
  const jumpToPageRef = useRef<(page: number) => void>(() => {});

  const handleJumpToPage = useCallback((page: number) => {
    jumpToPageRef.current?.(page);
  }, []);

  useEffect(() => {
    if (pdfs.length > 0 && !selectedPDF) {
      setSelectedPDF(pdfs[0]);
    }
  }, [pdfs, selectedPDF]);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setUploadStatus({ message: 'PDF Uploading...', type: 'uploading' });
    setIsLoading(true);
    
    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('files', file));

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/upload_pdfs', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const newPdfs = response.data.files || [];
      setPdfs(prev => [...prev, ...newPdfs]);

      if (response.data.summary) {
        setMessages(prev => [...prev, { 
          text: response.data.summary, 
          isUser: false, 
          type: 'summary' 
        }]);
      }

      if (newPdfs.length > 0 && !selectedPDF) {
        setSelectedPDF(newPdfs[0]);
      }

      setUploadStatus({ message: 'PDF Uploaded Successfully!', type: 'success' });
      setTimeout(() => setUploadStatus(null), 5000);
    } catch (error) {
      console.error('Upload failed:', error);
      setMessages(prev => [...prev, { 
        text: 'Failed to upload PDFs', 
        isUser: false, 
        type: 'error' 
      }]);
      setUploadStatus(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAgentClick = async (agent: string) => {
    setMessages(prev => [...prev, { 
      text: `${agent}?`, 
      isUser: true, 
      type: 'question' 
    }]);

    let endpoint, type;
    switch (agent) {
      case 'Summarize': endpoint = '/api/summarize'; type = 'summary'; break;
      case 'Abstract': endpoint = '/api/abstract'; type = 'abstract'; break;
      case 'Key Findings': endpoint = '/api/key_findings'; type = 'key_findings'; break;
      case 'Challenges': endpoint = '/api/challenges'; type = 'challenges'; break;
      default: return;
    }

    try {
      setIsLoading(true);
      const response = await axios.post(`http://127.0.0.1:8000${endpoint}`);
      setMessages(prev => [...prev, { 
        text: response.data[type] || 'No response', 
        isUser: false, 
        type 
      }]);
    } catch (error) {
      console.error(`Agent ${agent} failed:`, error);
      setMessages(prev => [...prev, { 
        text: `Error with ${agent}`, 
        isUser: false, 
        type: 'error' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async (text: string) => {
    if (!text.trim()) return;

    setMessages(prev => [...prev, { text, isUser: true, type: 'question' }]);

    try {
      setIsLoading(true);
      const response = await axios.post('http://127.0.0.1:8000/api/query', { 
        query: text, 
        pdf_name: selectedPDF 
      });

      const answer = response.data.answer || 'No answer';
      const citations = response.data.citations || '';
      const pageNumbers = [...citations.matchAll(/page=(\d+)/g)].map(m => parseInt(m[1], 10));
      const citationText = pageNumbers.length > 0
        ? `\n\nCitations: ${pageNumbers.map(page => `[Page ${page}]`).join(', ')}`
        : '';
      
      setMessages(prev => [...prev, { 
        text: answer + citationText, 
        isUser: false, 
        type: 'answer' 
      }]);
    } catch (error) {
      console.error('Query failed:', error);
      setMessages(prev => [...prev, { 
        text: 'Error processing query', 
        isUser: false, 
        type: 'error' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      {uploadStatus && (
        <div className={`${styles.uploadStatus} ${
          uploadStatus.type === 'uploading' 
            ? styles.uploadingStatus 
            : styles.successStatus
        }`}>
          {uploadStatus.message}
        </div>
      )}

      <div className={styles.container}>
        <div className={styles.pdfUploadSection}>
          <div className={styles.uploadContainer}>
            <label 
              htmlFor="pdf-upload" 
              className={`${styles.uploadButton} ${
                isLoading && uploadStatus?.type === 'uploading' 
                  ? styles.uploadingButton 
                  : uploadStatus?.type === 'success' 
                    ? styles.successButton 
                    : ''
              }`}
            >
              {isLoading && uploadStatus?.type === 'uploading' ? 'Uploading...' : 'Upload PDF'}
            </label>
            <input
              id="pdf-upload"
              type="file"
              multiple
              onChange={handleUpload}
              accept="application/pdf"
              className={styles.uploadInput}
              disabled={isLoading}
            />
          </div>
          
          {pdfs.length > 0 && (
            <div className={styles.pdfList}>
              {pdfs.map((file, index) => (
                <div
                  key={index}
                  className={`${styles.pdfItem} ${
                    selectedPDF === file ? styles.active : ''
                  }`}
                  onClick={() => setSelectedPDF(file)}
                >
                  {file}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className={styles.pdfPreview}>
          <PDFPreview
            pdfName={selectedPDF}
            onJumpToPageReady={(jumpFn) => {
              jumpToPageRef.current = jumpFn;
            }}
            isUploading={isLoading && uploadStatus?.type === 'uploading'}
          />
        </div>

        <div className={styles.chatArea}>
          <ChatWindow
            messages={messages}
            onAgentClick={handleAgentClick}
            onSend={handleSend}
            jumpToPage={handleJumpToPage}
            isSending={isLoading}
          />
        </div>
      </div>
    </Layout>
  );
}