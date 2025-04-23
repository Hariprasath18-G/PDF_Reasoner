from autogen import AssistantAgent, UserProxyAgent
from .config import config
from .rag_pipeline import RAGPipeline
from .embedding import EmbeddingModel
from .vector_store import VectorStore
import logging
import re
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AutoGen configuration for Gemma3-27B
llm_config = {
    "model": config.AUTOGEN_MODEL,
    "api_key": config.AUTOGEN_API_KEY,
    "base_url": config.AUTOGEN_API_BASE,
    "api_type": "openai",
    "timeout": config.GEMMA_API_TIMEOUT
}

def create_agents():
    """Create AutoGen agents for different tasks."""
    logger.info(f"Creating agents with llm_config: {llm_config}")
    user_proxy = UserProxyAgent(
        name="UserProxy",
        system_message="A proxy for the user to initiate tasks and receive one response.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        is_termination_msg=lambda x: True
    )

    summarizer = AssistantAgent(
        name="Summarizer",
        system_message="You are an expert at summarizing academic papers. Using the provided context, produce exactly 5 bullet points starting with '-' (e.g., - Objective:). Focus on the paper's objectives, methodology, findings, and contributions. Use plain text with no markdown formatting like '', '', '#', or '##'. Do not include any preamble, introduction, or explanation such as 'Here is a summary' or 'This is a 5-bullet point summary'. Do not comment on other summaries or citations. Use only the provided context. Example of incorrect output: 'Here is a summary: * **Objective:* Study X.'",
        llm_config=llm_config
    )

    abstractor = AssistantAgent(
        name="Abstractor",
        system_message="You are an expert at writing abstracts for academic papers. Using the provided context, write a 200-250 word abstract covering the problem statement, proposed methodology, key findings, and relevant metrics. Use plain text with no markdown formatting like '', '', '#', or '##'. Do not include any preamble, introduction, or explanation such as 'Here is an abstract' or 'This is a summary'. Use only the provided context. Example of incorrect output: 'Here is an abstract: * **Problem:* X.'",
        llm_config=llm_config
    )

    key_findings = AssistantAgent(
        name="KeyFindings",
        system_message="You are an expert at identifying key findings in academic papers. Using the provided context, produce exactly 5 bullet points starting with '-' (e.g., - Finding:). Focus on the most significant results and insights, avoiding speculation or minor details. Use plain text with no markdown formatting like '', '*', '#', or '##'. Do not include any preamble, introduction, or explanation such as 'Here are the findings' or 'This is a 5-bullet point list'. Use only the provided context.",
        llm_config=llm_config
    )

    challenges = AssistantAgent(
        name="Challenges",
        system_message="You are an expert at identifying challenges in academic papers. Search the provided context to identify up to 5 specific challenges or limitations mentioned in the paper, presenting them as bullet points starting with '-' (e.g., - Challenge:). Use plain text with no markdown formatting like '', '*', '#', or '##'. Do not include any preamble, introduction, or explanation such as 'Here are the challenges' or 'This is a list of challenges'. If no challenges are found, output '- No challenges identified'. Use only the provided context.",
        llm_config=llm_config
    )

    return {
        "user_proxy": user_proxy,
        "summarizer": summarizer,
        "abstractor": abstractor,
        "key_findings": key_findings,
        "challenges": challenges
    }

def run_agent(agent, user_proxy, context: str, task: str, page_numbers: List[int], chunks: List[Tuple[str, float, int, str]]) -> Tuple[str, List[dict]]:
    """Run an agent with the given context, task, and chunks, returning response and citations."""
    try:
        logger.info(f"Running agent {agent.name} with task: {task[:100]}...")
        logger.debug(f"Full context sent to agent: {context[:500]}...")
        logger.info(f"Chunk results: {[(c[0][:50], c[2], c[3]) for c in chunks]}")
        user_proxy.initiate_chat(
            agent,
            message=f"Context: {context}\nTask: {task}"
        )
        response = agent.last_message().get("content", "No response from agent")
        
        # Post-process to remove unwanted markdown and preambles
        response = re.sub(r'\\([^\s*]+)\*', r'\1', response)  # Remove backslash and asterisk around text
        response = re.sub(r'\*([^\s*]+)\*', r'\1', response)  # Remove asterisks around text
        response = re.sub(r'^#+.*\n', '', response)  # Remove markdown headers
        if agent.name == "Summarizer":
            response = re.sub(r'^Here is a [^\n]*summary[^\n]*:\n+', '', response, flags=re.IGNORECASE)
            lines = response.split('\n')
            response = '\n'.join(
                f"- {line.strip()[2:]}" if line.strip().startswith('*') else line
                for line in lines if line.strip()
            )
        elif agent.name == "Abstractor":
            response = re.sub(r'^Here is a [^\n]*abstract[^\n]*:\n+', '', response, flags=re.IGNORECASE)
        elif agent.name == "KeyFindings":
            response = re.sub(r'^Here is a [^\n]*summary[^\n]*:\n+', '', response, flags=re.IGNORECASE)
            lines = response.split('\n')
            response = '\n'.join(
                f"- {line.strip()[2:]}" if line.strip().startswith('*') else line
                for line in lines if line.strip()
            )
        elif agent.name == "Challenges":
            response = re.sub(r'^Here is a [^\n]*summary[^\n]*:\n+', '', response, flags=re.IGNORECASE)
            lines = response.split('\n')
            response = '\n'.join(
                f"- {line.strip()[2:]}" if line.strip().startswith('*') else line
                for line in lines if line.strip()
            )
            
        # Remove extra newlines
        response = re.sub(r'\n{3,}', r'\n\n', response).strip()
        
        # Generate citations from chunks
        citations = []
        if chunks:
            relevant_chunks = sorted(chunks, key=lambda x: x[1])[:3]
            citations = [{"pdf": chunk[3], "page": chunk[2]} for chunk in relevant_chunks]
        
        logger.info(f"Agent {agent.name} processed response: {response[:100]}... with citations: {citations}")
        return response, citations
    except Exception as e:
        logger.error(f"AutoGen error for {agent.name}: {str(e)}", exc_info=True)
        logger.info("Falling back to RAG pipeline query_gemma")
        try:
            rag = RAGPipeline(EmbeddingModel(config.EMBEDDING_MODEL), VectorStore(config.FAISS_INDEX_PATH))
            prompt = f"Context: {context}\nTask: {task}"
            response = rag.query_gemma(prompt)
            response = re.sub(r'\\([^\s*]+)\*', r'\1', response)  # Remove backslash and asterisk around text
            response = re.sub(r'\*([^\s*]+)\*', r'\1', response)  # Remove asterisks around text
            response = re.sub(r'^#+.*\n', '', response)  # Remove markdown headers
            if agent.name == "Summarizer":
                response = re.sub(r'^Here is a [^\n]*summary[^\n]*:\n+', '', response, flags=re.IGNORECASE)
                lines = response.split('\n')
                response = '\n'.join(
                    f"- {line.strip()[2:]}" if line.strip().startswith('*') else line
                    for line in lines if line.strip()
                )
            elif agent.name == "Abstractor":
                response = re.sub(r'^Here is a [^\n]*abstract[^\n]*:\n+', '', response, flags=re.IGNORECASE)
            elif agent.name == "KeyFindings":
                response = re.sub(r'^Here is a [^\n]*summary[^\n]*:\n+', '', response, flags=re.IGNORECASE)
                lines = response.split('\n')
                response = '\n'.join(
                    f"- {line.strip()[2:]}" if line.strip().startswith('*') else line
                    for line in lines if line.strip()
                )
            elif agent.name == "Challenges":
                response = re.sub(r'^Here is a [^\n]*summary[^\n]*:\n+', '', response, flags=re.IGNORECASE)
                lines = response.split('\n')
                response = '\n'.join(
                    f"- {line.strip()[2:]}" if line.strip().startswith('*') else line
                    for line in lines if line.strip()
                )
            
            response = re.sub(r'\n{3,}', r'\n\n', response).strip()
            
            # Generate citations for fallback
            citations = []
            if chunks:
                relevant_chunks = sorted(chunks, key=lambda x: x[1])[:3]
                citations = [{"pdf": chunk[3], "page": chunk[2]} for chunk in relevant_chunks]
            
            logger.info(f"Fallback response: {response[:100]}... with citations: {citations}")
            return response, citations
        except Exception as fallback_e:
            logger.error(f"Fallback error: {str(fallback_e)}")
            return f"Error running agent {agent.name}: {str(e)}. Fallback failed: {str(fallback_e)}", []