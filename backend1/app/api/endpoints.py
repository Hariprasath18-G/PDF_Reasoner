from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import faiss
from ..pdf_processor import process_pdfs
from ..embedding import EmbeddingModel
from ..vector_store import VectorStore
from ..rag_pipeline import RAGPipeline
from ..agents import create_agents, run_agent
from ..config import config
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Define a directory for storing PDFs
PDF_STORAGE_DIR = "uploaded_pdfs"
if not os.path.exists(PDF_STORAGE_DIR):
    os.makedirs(PDF_STORAGE_DIR)

# Initialize components
embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
vector_store = VectorStore(config.FAISS_INDEX_PATH)
rag_pipeline = RAGPipeline(embedding_model, vector_store)
agents = create_agents()

class QueryRequest(BaseModel):
    query: str

@router.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process multiple PDFs, then run summarizer agent."""
    try:
        vector_store.reset()
        logger.info("Reset FAISS index before processing new PDFs")
        
        pdf_paths = []
        pdf_names = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            file_path = os.path.join(PDF_STORAGE_DIR, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            pdf_paths.append(file_path)
            pdf_names.append(file.filename)
            logger.info(f"Attempted to save file: {file_path} with size {len(content)} bytes")
            if not os.path.exists(file_path):
                logger.error(f"Failed to save file: {file_path}")
                raise HTTPException(status_code=500, detail=f"Failed to save {file.filename}")
            else:
                logger.info(f"Verified file exists at: {file_path} with size {os.path.getsize(file_path)} bytes")
                # Debug: Check saved file content
                with open(file_path, "rb") as f_check:
                    first_bytes = f_check.read(10)
                    logger.info(f"First 10 bytes of saved file: {first_bytes.hex()}")

        chunks = process_pdfs(pdf_paths, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            logger.error("No chunks extracted from PDFs")
            raise HTTPException(status_code=400, detail="No text extracted from the provided PDFs")
        
        logger.debug(f"Sample chunk: {chunks[0][0][:100]}... from page {chunks[0][1]} of {chunks[0][2]}")
        embeddings = embedding_model.encode([chunk[0] for chunk in chunks])
        vector_store.add_vectors(embeddings, chunks)
        
        context, page_numbers, chunk_results = rag_pipeline.get_full_context(pdf_name=pdf_names[0], agent_name="Summarizer")
        logger.info(f"Summary context length: {len(context)}, pages: {page_numbers}")
        if not context:
            logger.warning("No content available for summarization")
            return {
                "message": f"Successfully processed {len(files)} PDFs with {len(chunks)} chunks",
                "files": pdf_names,
                "summary": "No content available for summarization.",
                "citations": []
            }
        
        summary, citations = run_agent(
            agents["summarizer"],
            agents["user_proxy"],
            context,
            "Produce 5 bullet points starting with '-' focusing on objectives, methodology, findings, and contributions.",
            page_numbers,
            chunk_results
        )
        
        return {
            "message": f"Successfully processed {len(files)} PDFs with {len(chunks)} chunks",
            "files": pdf_names,
            "summary": summary,
            "citations": citations
        }
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@router.post("/query")
async def query_pdf(request: QueryRequest):
    """Query the processed PDFs."""
    try:
        answer, page_numbers, chunks = rag_pipeline.generate_answer(request.query, pdf_name=None)
        citations = [{"pdf": chunk[3], "page": chunk[2]} for chunk in sorted(chunks, key=lambda x: x[1])[:3]] if chunks else []
        return {"answer": answer, "citations": citations}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/reset_index")
async def reset_index():
    """Reset the FAISS index and text storage."""
    try:
        vector_store.reset()
        return {"message": "FAISS index reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting index: {str(e)}")

@router.post("/abstract")
async def generate_abstract():
    """Generate an abstract using the abstractor agent."""
    try:
        context, page_numbers, chunk_results = rag_pipeline.get_full_context(pdf_name=None, agent_name="Abstractor")
        logger.info(f"Abstract context length: {len(context)}, pages: {page_numbers}")
        if not context:
            logger.warning("No content available for abstract")
            return {"abstract": "No content available for abstract generation.", "citations": []}
        abstract, citations = run_agent(
            agents["abstractor"],
            agents["user_proxy"],
            context,
            "Write a 200-250 word abstract covering the problem statement, proposed methodology, key findings, and metrics.",
            page_numbers,
            chunk_results
        )
        return {"abstract": abstract, "citations": citations}
    except Exception as e:
        logger.error(f"Error generating abstract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating abstract: {str(e)}")

@router.post("/key_findings")
async def generate_key_findings():
    """Generate key findings using the key_findings agent."""
    try:
        context, page_numbers, chunk_results = rag_pipeline.get_full_context(pdf_name=None, agent_name="KeyFindings")
        logger.info(f"Key findings context length: {len(context)}, pages: {page_numbers}")
        if not context:
            logger.warning("No content available for key findings")
            return {"key_findings": "No content available for key findings generation.", "citations": []}
        findings, citations = run_agent(
            agents["key_findings"],
            agents["user_proxy"],
            context,
            "Produce 5 bullet points starting with '-' focusing on the most significant results and insights.",
            page_numbers,
            chunk_results
        )
        return {"key_findings": findings, "citations": citations}
    except Exception as e:
        logger.error(f"Error generating key findings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating key findings: {str(e)}")

@router.post("/challenges")
async def generate_challenges():
    """Generate challenges using the challenges agent."""
    try:
        context, page_numbers, chunk_results = rag_pipeline.get_full_context(pdf_name=None, agent_name="Challenges")
        logger.info(f"Challenges context length: {len(context)}, pages: {page_numbers}")
        if not context:
            logger.warning("No content available for challenges")
            return {"challenges": "No content available for challenges generation.", "citations": []}
        challenges, citations = run_agent(
            agents["challenges"],
            agents["user_proxy"],
            context,
            "Identify up to 5 specific challenges or limitations mentioned in the paper, presenting them as bullet points starting with '-' (e.g., - Challenge:).",
            page_numbers,
            chunk_results
        )
        return {"challenges": challenges, "citations": citations}
    except Exception as e:
        logger.error(f"Error generating challenges: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating challenges: {str(e)}")

@router.get("/get_pdf")
async def get_pdf(pdf_name: str):
    """Serve the uploaded PDF."""
    file_path = os.path.join(PDF_STORAGE_DIR, pdf_name)
    logger.info(f"Received request for PDF: {pdf_name}")
    logger.info(f"Checking file at: {file_path}")
    if os.path.exists(file_path):
        logger.info(f"File found, size: {os.path.getsize(file_path)} bytes, serving: {file_path}")
        return FileResponse(file_path, media_type="application/pdf", filename=pdf_name, headers={"Content-Disposition": f"inline; filename={pdf_name}"})
    logger.error(f"File not found at: {file_path}. Directory contents: {os.listdir(PDF_STORAGE_DIR)}")
    raise HTTPException(status_code=404, detail=f"PDF not found at {file_path}")