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

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API Keys and Authentication
# ============================================================================

# Check for required API keys
if "GROQ_API_KEY" not in os.environ:
    raise MissingAPIKey("Groq LLM")

GROQ_API_KEY = os.environ["GROQ_API_KEY"]


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

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_VECTOR_SIZE = 768

# Thread pool for CPU-intensive operations
MAX_WORKERS = 4


# ============================================================================
# Initialize Models and Resources
# ============================================================================

# Initialize Groq LLM
llm = ChatGroq(
    temperature=LLM_TEMPERATURE, 
    model_name=LLM_MODEL_NAME,
    api_key=GROQ_API_KEY
)

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
    "Custom Error Handling"
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
        "features": APP_FEATURES
    }