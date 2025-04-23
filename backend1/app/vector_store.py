import faiss
import numpy as np
import json
import os
import logging
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_path: str, dimension: int = 384):
        self.index_path = index_path
        self.texts_path = f"{index_path}_texts.json"
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        self.page_numbers = []
        self.pdf_names = []
        
        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.texts_path, 'r') as f:
                    data = json.load(f)
                    self.texts = data.get("texts", [])
                    self.page_numbers = data.get("page_numbers", [])
                    self.pdf_names = data.get("pdf_names", [])
                if not all(isinstance(p, int) and p > 0 for p in self.page_numbers):
                    logger.error("Invalid page numbers detected, resetting index")
                    self.reset()
                elif len(set(self.page_numbers)) == 1 and self.page_numbers:
                    logger.warning(f"All chunks assigned to page {self.page_numbers[0]}, possible error")
                logger.info(f"Loaded FAISS index with {len(self.texts)} vectors, pages: {sorted(set(self.page_numbers))}, PDFs: {set(self.pdf_names)}")
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                self.reset()
    
    def add_vectors(self, vectors: np.ndarray, texts: List[Tuple[str, int, str]]):
        """Add vectors, texts, page numbers, and PDF names to the index."""
        try:
            if not vectors.shape[1] == self.dimension:
                raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")
            
            self.index.add(vectors)
            new_texts, new_pages, new_pdfs = zip(*texts)
            self.texts.extend(new_texts)
            self.page_numbers.extend(new_pages)
            self.pdf_names.extend(new_pdfs)
            
            faiss.write_index(self.index, self.index_path)
            with open(self.texts_path, 'w') as f:
                json.dump({
                    "texts": self.texts,
                    "page_numbers": self.page_numbers,
                    "pdf_names": self.pdf_names
                }, f)
            logger.info(f"Added {len(texts)} vectors to the index. Pages: {sorted(set(self.page_numbers))}, PDFs: {set(self.pdf_names)}")
        except Exception as e:
            logger.error(f"Error adding vectors: {str(e)}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 10, threshold: float = 0.3, pdf_name: str = None) -> List[Tuple[str, float, int, str]]:
        """Search the index for similar vectors, optionally filtering by PDF name."""
        try:
            logger.info(f"Available pdf_names: {set(self.pdf_names)}")
            distances, indices = self.index.search(query_vector, k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.texts) and distances[0][i] < threshold:
                    if pdf_name is None or self.pdf_names[idx] == pdf_name:
                        results.append((self.texts[idx], distances[0][i], self.page_numbers[idx], self.pdf_names[idx]))
            logger.info(f"Search returned {len(results)} results for pdf_name: {pdf_name}, pages: {[r[2] for r in results]}")
            return results
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            return []
    
    def reset(self):
        """Reset the index and clear stored texts, page numbers, and PDF names."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.page_numbers = []
        self.pdf_names = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.texts_path):
            os.remove(self.texts_path)
        logger.info("Vector store reset successfully")