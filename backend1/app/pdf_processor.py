import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
import logging
import pytesseract
from PIL import Image
import io
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file, with OCR fallback for image-based PDFs."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            logger.debug(f"Page {page_num + 1} text: {page_text[:100]}...")
            if not page_text.strip():
                logger.info(f"No text extracted from page {page_num + 1}, attempting OCR")
                pix = page.get_pixmap(dpi=300)  # Increase DPI for better OCR
                img = Image.open(io.BytesIO(pix.tobytes()))
                page_text = pytesseract.image_to_string(img, config='--psm 6')  # Assume block text
                logger.info(f"OCR extracted {len(page_text)} characters from page {page_num + 1}")
            text += page_text + "\n"
        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF: {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        raise

def process_pdfs(pdf_paths: List[str], chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, str]]:
    """Process multiple PDFs into chunks with page numbers and PDF identifiers."""
    try:
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        for pdf_path in pdf_paths:
            logger.info(f"Processing PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            pdf_name = os.path.basename(pdf_path).replace("temp_", "")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                logger.debug(f"Page {page_num + 1} content: {page_text[:100]}...")
                if not page_text.strip():
                    logger.info(f"No text on page {page_num + 1}, attempting OCR")
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    page_text = pytesseract.image_to_string(img, config='--psm 6')
                    logger.info(f"OCR extracted {len(page_text)} characters from page {page_num + 1}")
                
                if not page_text.strip():
                    logger.warning(f"No content extracted from page {page_num + 1} of {pdf_path}")
                    continue
                
                chunks = text_splitter.split_text(page_text)
                for chunk in chunks:
                    all_chunks.append((chunk, page_num + 1, pdf_name))
                    logger.debug(f"Chunk from page {page_num + 1} of {pdf_name}: {chunk[:100]}...")
            
            doc.close()
            logger.info(f"Generated {len(chunks)} chunks from PDF: {pdf_path}")
        
        if not all_chunks:
            logger.error("No chunks generated from any PDFs")
            raise ValueError("No text extracted from the provided PDFs")
        
        page_counts = {chunk[2]: sorted(set(chunk[1] for chunk in all_chunks if chunk[2] == chunk[2])) for chunk in all_chunks}
        logger.info(f"Page numbers per PDF: {page_counts}")
        
        logger.info(f"Processed {len(pdf_paths)} PDFs into {len(all_chunks)} chunks")
        return all_chunks
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        raise