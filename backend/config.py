"""
Configuration management for RAG System.
Centralizes all settings, environment variables, and model initialization.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from exceptions import MissingAPIKey

# Attempt to import Ollama, required for local LLM support
try:
    from langchain_community.llms import Ollama
    from requests.exceptions import ConnectionError
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("WARNING: Ollama is not installed or dependencies are missing. Only Groq will be available.")
except ConnectionError:
    # This might trigger later, but useful to catch here if possible
    OLLAMA_AVAILABLE = False
    print("WARNING: Cannot connect to Ollama. Falling back to Groq.")

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API Keys and Authentication
# ============================================================================

# Check for required API keys
#if "GROQ_API_KEY" not in os.environ:
#    raise MissingAPIKey("Groq LLM")

#GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# General Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower() # Default to ollama
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check for required API keys based on provider priority
if LLM_PROVIDER == "groq" or (LLM_PROVIDER == "ollama" and not OLLAMA_AVAILABLE):
    if not GROQ_API_KEY:
        raise MissingAPIKey("Groq LLM")

# ============================================================================
# Storage Configuration
# ============================================================================

QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "./qdrant_storage")
COLLECTION_NAME = "all_documents"


# ============================================================================
# File Upload Configuration
# ============================================================================

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
TEMP_UPLOAD_DIR = "temp"

# Supported file extensions
SUPPORTED_EXTENSIONS = [
    '.pdf', '.doc', '.docx', '.txt', '.csv', '.xlsx', 
    '.xls', '.pptx', '.ppt', '.html', '.htm', '.md', 
    '.json', '.xml'
]

# File type mapping
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


# ============================================================================
# Document Processing Configuration
# ============================================================================

# Text chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 20

# Text splitter separators
TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ============================================================================
# Retrieval Configuration
# ============================================================================

# Number of documents to retrieve
RETRIEVAL_K = 10

# Ensemble retriever weights [BM25, Vector]
ENSEMBLE_WEIGHTS = [0.4, 0.6]

# Conversation history length
MAX_CONVERSATION_HISTORY = 20


# ============================================================================
# Model Configuration
# ============================================================================

# LLM settings
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0

# Ollama Settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# NOTE: Using host.docker.internal to access Windows host from Docker container

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_VECTOR_SIZE = 768

# Thread pool for CPU-intensive operations
MAX_WORKERS = 4


# ============================================================================
# Initialize Models and Resources
# ============================================================================

# Initialize Groq LLM
#llm = ChatGroq(
#    temperature=LLM_TEMPERATURE, 
#    model_name=LLM_MODEL_NAME,
#    api_key=GROQ_API_KEY
#)

# ============================================================================
# Model Initialization
# ============================================================================

def initialize_llm():
    """Initializes the LLM with Ollama as primary and Groq as fallback."""
    global LLM_PROVIDER
    
    # 1. Attempt Ollama (Primary)
    if LLM_PROVIDER == "ollama" and OLLAMA_AVAILABLE:
        try:
            llm_instance = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=LLM_TEMPERATURE
            )
            # A quick test call to ensure the service is actually running
            # Ollama needs to be accessible from the Docker container
            llm_instance.invoke("test connectivity", config={"timeout": 5}) 
            
            print(f"INFO: Successfully initialized Ollama LLM: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
            return llm_instance
        except (ConnectionError, Exception) as e:
            print(f"WARNING: Failed to connect to Ollama at {OLLAMA_BASE_URL} ({type(e).__name__}). Falling back to Groq.")
            # Fall through to Groq initialization

    # 2. Initialize Groq (Secondary / Explicit choice)
    if GROQ_API_KEY:
        llm_instance = ChatGroq(
            temperature=LLM_TEMPERATURE,
            model_name=LLM_MODEL_NAME,
            api_key=GROQ_API_KEY
        )
        # Update the provider variable for runtime tracking
        LLM_PROVIDER = "groq"
        print(f"INFO: Using Groq LLM (Fallback or Explicit): {LLM_MODEL_NAME}")
        return llm_instance
    else:
        # If Ollama failed and Groq key is missing, this is a fatal configuration error
        raise MissingAPIKey("Groq LLM (Ollama fallback failed or not selected)")

# Initialize LLM (Must be called after all variables are set)
llm = initialize_llm()

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


# ============================================================================
# Application Metadata
# ============================================================================

APP_TITLE = "Enhanced Multi-File RAG System with Persistence"
APP_DESCRIPTION = "High-performance RAG system with persistent storage and concurrent user support"
APP_VERSION = "2.1.0"

APP_FEATURES = [
    "Persistent Storage",
    "Concurrent Processing", 
    "Single File Delete",
    "Performance Monitoring",
    "Custom Error Handling",
    "Ollama (Local LLM) Support" # Added new feature
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_storage_info() -> dict:
    """Get storage configuration information."""
    return {
        "type": "persistent",
        "path": QDRANT_STORAGE_PATH,
        "collection": COLLECTION_NAME
    }


def get_app_info() -> dict:
    """Get application information."""
    return {
        "title": APP_TITLE,
        "version": APP_VERSION,
        "features": APP_FEATURES,
        "llm_provider": LLM_PROVIDER, # Use the potentially updated global variable 
        "llm_model": OLLAMA_MODEL if LLM_PROVIDER == "ollama" else LLM_MODEL_NAME, 
        "embedding_model": EMBEDDING_MODEL_NAME
    }