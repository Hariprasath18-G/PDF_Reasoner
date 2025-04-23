import styles from './PDFList.module.css';

type PDFListProps = {
  pdfs: string[];
  onSelect: (pdf: string) => void;
};

export default function PDFList({ pdfs, onSelect }: PDFListProps) {
  return (
    <div className={styles.pdfList}>
      {pdfs.map((pdf, index) => (
        <div
          key={index}
          className={styles.pdfItem}
          onClick={() => {
            console.log('Selected PDF:', pdf); // Debug log
            onSelect(pdf);
          }}
        >
          {pdf}
        </div>
      ))}
    </div>
  );
}