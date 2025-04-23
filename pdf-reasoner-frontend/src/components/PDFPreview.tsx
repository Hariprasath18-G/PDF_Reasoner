import { useState, useEffect } from 'react';
import { Worker, Viewer } from '@react-pdf-viewer/core';
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout';
import { pageNavigationPlugin } from '@react-pdf-viewer/page-navigation';
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';
import '@react-pdf-viewer/page-navigation/lib/styles/index.css';
import styles from '../styles/Home.module.css';

interface PdfPreviewProps {
  pdfName: string | null;
  onJumpToPageReady: (jumpFn: (page: number) => void) => void;
  isUploading?: boolean;
}

const PdfPreviewComponent = ({ pdfName, onJumpToPageReady, isUploading = false }: PdfPreviewProps) => {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const pageNavigationPluginInstance = pageNavigationPlugin();
  const defaultLayoutPluginInstance = defaultLayoutPlugin();

  useEffect(() => {
    if (pageNavigationPluginInstance.jumpToPage) {
      onJumpToPageReady((page: number) => {
        pageNavigationPluginInstance.jumpToPage(page - 1);
      });
    }
  }, [onJumpToPageReady, pageNavigationPluginInstance]);

  useEffect(() => {
    if (!pdfName) {
      setPdfUrl(null);
      setError(null);
      return;
    }

    const fetchPdf = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/api/get_pdf?pdf_name=${encodeURIComponent(pdfName)}`,
          { headers: { 'Accept': 'application/pdf' } }
        );

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const blob = await response.blob();
        setPdfUrl(URL.createObjectURL(blob));
        setError(null);
      } catch (err) {
        console.error('PDF fetch error:', err);
        setError(`Failed to load PDF: ${err instanceof Error ? err.message : String(err)}`);
        setPdfUrl(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPdf();
  }, [pdfName]);

  if (error) return <div className={styles.pdfPreviewContainer}>{error}</div>;

  return (
    <div className={styles.pdfPreviewContainer}>
      {(!pdfUrl && !isUploading) && (
        <div className={`${styles.noPdfSelected} ${pdfName ? styles.hidden : ''}`}>
          <p>No PDF selected</p>
          <p>Upload a PDF to get started</p>
        </div>
      )}

      {(isUploading || isLoading) && (
        <div className={styles.pdfLoadingOverlay}>
          <div className={styles.loadingSpinner} />
          <p>{isUploading ? 'Uploading PDF...' : 'Loading PDF...'}</p>
        </div>
      )}

      {pdfUrl && (
        <Worker workerUrl="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js">
          <Viewer
            fileUrl={pdfUrl}
            plugins={[defaultLayoutPluginInstance, pageNavigationPluginInstance]}
          />
        </Worker>
      )}
    </div>
  );
};

export default PdfPreviewComponent;