import requests
import re
from typing import List, Tuple
from .config import config
from .embedding import EmbeddingModel
from .vector_store import VectorStore
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def query_gemma(self, prompt: str) -> str:
        """Query Gemma3-27B API with a prompt."""
        headers = {
            "Authorization": f"Bearer {config.GEMMA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gemma3-27b",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500
        }
        
        try:
            logger.info(f"Sending request to {config.GEMMA_API_URL} with payload: {payload}")
            response = requests.post(config.GEMMA_API_URL, json=payload, headers=headers, timeout=config.GEMMA_API_TIMEOUT)
            response.raise_for_status()
            
            response_data = response.json()
            logger.info(f"Received response: {response_data}")
            
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response from Gemma")
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return f"Error querying Gemma: HTTP {response.status_code} - {response.text}"
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
            return "Error querying Gemma: Failed to connect to the API server"
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return f"Error querying Gemma: Request timed out after {config.GEMMA_API_TIMEOUT} seconds"
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
            return f"Error querying Gemma: {str(req_err)}"
        except ValueError as val_err:
            logger.error(f"JSON decode error: {val_err}")
            return "Error querying Gemma: Invalid response format from API"
    
    def generate_answer(self, query: str, pdf_name: str = None, k: int = 10) -> Tuple[str, List[int], List[Tuple[str, float, int, str]]]:
        """Generate an answer using RAG, returning the answer, page numbers, and chunks."""
        logger.info(f"Processing query: {query} for PDF: {pdf_name}")
        query_embedding = self.embedding_model.encode([query])
        results = self.vector_store.search(query_embedding, k, threshold=0.5, pdf_name=pdf_name)
        
        if not results:
            logger.warning(f"No relevant documents found for query: {query}")
            results = [(text, 0.0, page, name) for text, page, name in zip(
                self.vector_store.texts[:k], self.vector_store.page_numbers[:k], self.vector_store.pdf_names[:k]
            ) if pdf_name is None or name == pdf_name]
            logger.info(f"Fallback retrieved {len(results)} chunks")
        
        if not results:
            logger.warning("No chunks available even after fallback")
            return "No relevant information found in the processed PDFs.", [], []
        
        context = "\n".join([result[0] for result in results])
        page_numbers = [result[2] for result in results]
        logger.info(f"Retrieved {len(results)} chunks for context: {context[:200]}... with pages: {page_numbers}")
        
        prompt = f"""Context: {context}
Question: {query}
Answer the question based on the provided context. If the context doesn't contain enough information, say so."""
        
        answer = self.query_gemma(prompt)
        # Strip markdown
        answer = re.sub(r'\\([^\]+)\\*', r'\1', answer)
        answer = re.sub(r'\([^\]+)\*', r'\1', answer)
        answer = re.sub(r'^#+.*\n', '', answer)
        answer = re.sub(r'\n{3,}', r'\n\n', answer).strip()
        
        return answer, page_numbers, results
    
    def get_full_context(self, pdf_name: str = None, max_chars: int = 10000, agent_name: str = None) -> Tuple[str, List[int], List[Tuple[str, float, int, str]]]:
        """Retrieve relevant chunks from the vector store for agent context, with page numbers and chunks."""
        if not self.vector_store.texts:
            logger.warning("No texts available in vector store")
            return "", [], []
        
        # Use broader query based on agent name
        if agent_name == "ResultsDiscussion":
            query = "Performance metrics, errors, or quantitative results (e.g., MAE, RMSE, MAPE, accuracy, AUC, percentages)"
        elif agent_name == "KeyFindings":
            query = "Significant results and insights from the paper"
        elif agent_name == "Challenges":
            query = "Challenges, limitations, or issues mentioned in the paper"
        else:
            query = "Summarize the main objectives, methodology, findings, and contributions of the paper"
        
        query_embedding = self.embedding_model.encode([query])
        logger.info(f"Searching for context with pdf_name: {pdf_name}, query: {query}")
        results = self.vector_store.search(query_embedding, k=15, threshold=0.9, pdf_name=pdf_name)
        
        if not results:
            logger.warning(f"No relevant chunks found via search for pdf_name: {pdf_name}, falling back to all chunks")
            results = [(text, 0.0, page, name) for text, page, name in zip(
                self.vector_store.texts[:15], self.vector_store.page_numbers[:15], self.vector_store.pdf_names[:15]
            ) if pdf_name is None or name == pdf_name]
            logger.info(f"Fallback retrieved {len(results)} chunks: {[r[3] + ': Page ' + str(r[2]) for r in results]}")
        
        if not results:
            logger.warning("No chunks available even after fallback")
            return "", [], []
        
        context = "\n".join([result[0] for result in results])
        page_numbers = [result[2] for result in results]
        if len(context) > max_chars:
            context = context[:max_chars]
            page_numbers = page_numbers[:len(context.split('\n'))]
            logger.info(f"Truncated context to {max_chars} characters")
        logger.info(f"Retrieved context: {context[:200]}... with pages: {page_numbers}")
        return context, page_numbers, results