import os
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from exceptions import MissingAPIKey

logger = logging.getLogger(__name__)

# Try to import Ollama for local LLM support
try:
    from langchain_community.llms import Ollama
    from requests.exceptions import ConnectionError
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama is not installed or dependencies are missing. Only Groq will be available.")
except ConnectionError:
    OLLAMA_AVAILABLE = False
    logger.warning("Cannot connect to Ollama. Falling back to Groq.")

# Load env vars
load_dotenv()

# Debug mode for verbose logging and detailed error messages
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Enable debug endpoints (e.g., /debug/qdrant)
ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "false").lower() == "true"

# General config
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # default to ollama
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check for required API keys based on provider
if LLM_PROVIDER == "groq" or (LLM_PROVIDER == "ollama" and not OLLAMA_AVAILABLE):
    if not GROQ_API_KEY:
        raise MissingAPIKey("Groq LLM")

# Storage config
QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "./qdrant_storage")
COLLECTION_NAME = "all_documents"


# File upload settings
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
TEMP_UPLOAD_DIR = "temp"

SUPPORTED_EXTENSIONS = [
    '.pdf', '.doc', '.docx', '.txt', '.csv', '.xlsx', 
    '.xls', '.pptx', '.ppt', '.html', '.htm', '.md', 
    '.json', '.xml'
]

FILE_TYPE_MAP = {
    '.pdf': 'PDF',
    '.doc': 'Word',
    '.docx': 'Word', 
    '.txt': 'Text',
    '.csv': 'CSV',
    '.xlsx': 'Excel',
    '.xls': 'Excel',
    '.pptx': 'PowerPoint',
    '.ppt': 'PowerPoint',
    '.html': 'HTML',
    '.htm': 'HTML',
    '.md': 'Markdown',
    '.json': 'JSON',
    '.xml': 'XML'
}


# Text chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 20

# Text splitter separators
TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Number of docs to retrieve
RETRIEVAL_K = 10

# Ensemble retriever weights [BM25, Vector]
ENSEMBLE_WEIGHTS = [0.4, 0.6]

# Conversation history length
MAX_CONVERSATION_HISTORY = 20

# LLM settings
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_VECTOR_SIZE = 768

# Thread pool for CPU-intensive ops
MAX_WORKERS = 4

# Reranking config
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Docs to retrieve before reranking
RETRIEVAL_K_BEFORE_RERANK = 20

# Docs to keep after reranking
RETRIEVAL_K_AFTER_RERANK = 7

# Min relevance score (0.0 to 1.0)
MIN_RELEVANCE_SCORE = 0.3

# Contextual chunk enrichment
ENABLE_CONTEXTUAL_ENRICHMENT = os.getenv("ENABLE_CONTEXTUAL_ENRICHMENT", "true").lower() == "true"

# Context window size (chars before and after)
CONTEXT_WINDOW_CHARS = 200

# Include doc summary in each chunk
INCLUDE_DOC_SUMMARY = True

# Initialize reranker if enabled

reranker = None
if ENABLE_RERANKING:
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER_MODEL_NAME)
        logger.info(f"Reranker initialized: {RERANKER_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Failed to initialize reranker: {e}")
        logger.info("Continuing without reranking")
        ENABLE_RERANKING = False
        reranker = None

# Thread pool
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Model initialization

def initialize_llm():
    """Initialize LLM with Ollama as primary and Groq as fallback"""
    global LLM_PROVIDER
    
    # try Ollama first
    if LLM_PROVIDER == "ollama" and OLLAMA_AVAILABLE:
        try:
            llm_instance = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=LLM_TEMPERATURE
            )
            # quick test call to ensure service is running
            # Ollama needs to be accessible from Docker container
            llm_instance.invoke("test connectivity", config={"timeout": 5}) 
            
            logger.info(f"Successfully initialized Ollama LLM: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
            return llm_instance
        except (ConnectionError, Exception) as e:
            logger.warning(f"Failed to connect to Ollama at {OLLAMA_BASE_URL} ({type(e).__name__}). Falling back to Groq.")
            # fall through to Groq

    # use Groq as fallback
    if GROQ_API_KEY:
        llm_instance = ChatGroq(
            temperature=LLM_TEMPERATURE,
            model_name=LLM_MODEL_NAME,
            api_key=GROQ_API_KEY
        )
        # update provider variable for runtime tracking
        LLM_PROVIDER = "groq"
        logger.info(f"Using Groq LLM (Fallback or Explicit): {LLM_MODEL_NAME}")
        return llm_instance
    else:
        # if Ollama failed and Groq key is missing
        raise MissingAPIKey("Groq LLM (Ollama fallback failed or not selected)")

llm = initialize_llm()

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


# App metadata

APP_TITLE = "Enhanced Multi-File RAG System with Persistence"
APP_DESCRIPTION = "High-performance RAG system with persistent storage and concurrent user support"
APP_VERSION = "2.2.0"

APP_FEATURES = [
    "Persistent Storage",
    "Concurrent Processing", 
    "Single File Delete",
    "Performance Monitoring",
    "Custom Error Handling",
    "Ollama (Local LLM) Support",
    "Cross-Encoder Reranking" if ENABLE_RERANKING else None,
    "Contextual Chunk Enrichment" if ENABLE_CONTEXTUAL_ENRICHMENT else None
]

# remove None values
APP_FEATURES = [f for f in APP_FEATURES if f is not None]

# Helper functions

def get_storage_info() -> dict:
    """Get storage config info"""
    return {
        "type": "persistent",
        "path": QDRANT_STORAGE_PATH,
        "collection": COLLECTION_NAME
    }


def get_app_info() -> dict:
    """Get app information"""
    return {
        "title": APP_TITLE,
        "version": APP_VERSION,
        "features": APP_FEATURES,
        "llm_provider": LLM_PROVIDER,
        "llm_model": OLLAMA_MODEL if LLM_PROVIDER == "ollama" else LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranking_enabled": ENABLE_RERANKING,
        "reranker_model": RERANKER_MODEL_NAME if ENABLE_RERANKING else None,
        "contextual_enrichment_enabled": ENABLE_CONTEXTUAL_ENRICHMENT
    }