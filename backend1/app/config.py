from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
    GEMMA_API_URL = os.getenv("GEMMA_API_URL", "https://litellm.dev.ai-cloud.me/v1/chat/completions")
    GEMMA_API_TIMEOUT = int(os.getenv("GEMMA_API_TIMEOUT", 30))
    FAISS_INDEX_PATH = "faiss_index"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    AUTOGEN_MODEL = os.getenv("AUTOGEN_MODEL", "gemma3-27b")
    AUTOGEN_API_BASE = os.getenv("AUTOGEN_API_BASE", "https://litellm.dev.ai-cloud.me")
    AUTOGEN_API_KEY = os.getenv("GEMMA_API_KEY")

    def validate(self):
        if not self.GEMMA_API_KEY:
            raise ValueError("GEMMA_API_KEY is not set in .env")
        if not self.GEMMA_API_URL:
            raise ValueError("GEMMA_API_URL is not set in .env")
        if not self.AUTOGEN_API_BASE:
            raise ValueError("AUTOGEN_API_BASE is not set in .env")
        if not self.AUTOGEN_API_KEY:
            raise ValueError("AUTOGEN_API_KEY is not set in .env")

config = Config()
config.validate()