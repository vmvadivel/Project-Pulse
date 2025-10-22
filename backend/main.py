import os
import asyncio
import logging
from typing import Any, List
import time
from datetime import datetime

from exceptions import *
from config import *
from utils import *
from services import DocumentStore, process_document_sync

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Get debug mode from env
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Suppress DEBUG logs from all modules in production
if not DEBUG_MODE:
    # set root logger to INFO
    logging.getLogger().setLevel(logging.INFO)
    
    # silence noisy third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

logger.info(f"Application starting - Debug Mode: {DEBUG_MODE}")

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register custom error handlers
register_error_handlers(app)
logger.info("Custom error handlers registered")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

# Global document store
doc_store = DocumentStore()

class ChatRequest(BaseModel):
    query: str

class IngestResponse(BaseModel):
    message: str
    num_documents: int
    num_chunks: int
    total_files: int
    total_documents: int
    total_chunks: int
    file_type: str
    file_size_formatted: str
    processing_time: str
    contextual_enrichment_applied: bool

class FileListResponse(BaseModel):
    files: List[dict]
    total_files: int
    total_documents: int
    total_chunks: int

class DeleteFileResponse(BaseModel):
    message: str
    deleted_file: str
    remaining_files: int

class SystemStatsResponse(BaseModel):
    total_files: int
    total_documents: int
    total_chunks: int
    total_size_formatted: str
    avg_processing_time: str
    storage_type: str
    storage_path: str
    reranking_enabled: bool
    contextual_enrichment_enabled: bool

# API endpoints

@app.get("/",
    summary="System Information",
    description="Get system information, features, and configuration"
)
def read_root():
    """Root endpoint returning system info"""
    logger.debug("Root endpoint accessed")
    
    return {
        "message": APP_TITLE,
        "version": APP_VERSION,
        "features": APP_FEATURES,
        "storage": get_storage_info(),
        "llm_provider": LLM_PROVIDER,
        "reranking_enabled": ENABLE_RERANKING,
        "contextual_enrichment_enabled": ENABLE_CONTEXTUAL_ENRICHMENT
    }


@app.get("/health",
    summary="Health Check",
    description="Check system health and get basic statistics"
)
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns system status and basic stats.
    """
    stats = doc_store.get_stats()
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "reranking_enabled": ENABLE_RERANKING,
        "contextual_enrichment_enabled": ENABLE_CONTEXTUAL_ENRICHMENT
    }
    
    logger.debug(f"Health check - Files: {stats['total_files']}, Chunks: {stats['total_chunks']}")
    
    return health_data


@app.get("/stats",
    response_model=SystemStatsResponse,
    summary="System Statistics",
    description="Get detailed system statistics and configuration"
)
def get_system_stats():
    """
    Get system statistics including file counts, sizes, processing metrics,
    storage config, and feature flags
    """
    stats = doc_store.get_stats()
    
    logger.debug(f"Statistics requested - {stats['total_files']} files, {stats['total_chunks']} chunks")
    
    return SystemStatsResponse(
        total_files=stats['total_files'],
        total_documents=stats['total_documents'],
        total_chunks=stats['total_chunks'],
        total_size_formatted=format_file_size(stats['total_size']),
        avg_processing_time=f"{stats['avg_processing_time']:.2f}s",
        storage_type=stats['storage_type'],
        storage_path=stats['storage_path'],
        reranking_enabled=ENABLE_RERANKING,
        contextual_enrichment_enabled=ENABLE_CONTEXTUAL_ENRICHMENT
    )


@app.get("/files",
    response_model=FileListResponse,
    summary="List Files",
    description="Get list of all uploaded files with metadata"
)
def get_uploaded_files():
    """
    Get list of all uploaded files with metadata:
    filename, type, doc/chunk counts, size, upload date, processing time
    """
    files = doc_store.get_file_list()
    
    logger.debug(f"File list requested - {len(files)} files")
    
    return FileListResponse(
        files=files,
        total_files=len(files),
        total_documents=len(doc_store.all_documents),
        total_chunks=len(doc_store.all_texts)
    )


@app.delete("/files",
    summary="Clear All Files",
    description="Delete all uploaded files and reset the knowledge base"
)
def clear_all_files():
    """
    Clear all uploaded files and reset the system.
    
    WARNING: This is irreversible and will delete all docs, clear vector DB,
    reset conversation history, and remove all file metadata.
    """
    logger.warning("Clear all files requested")
    
    files_before = len(doc_store.file_metadata)
    doc_store.clear_all()
    
    logger.info(f"Cleared {files_before} files successfully")
    
    return {"message": "All files cleared successfully"}


@app.delete("/files/{filename}",
    response_model=DeleteFileResponse,
    summary="Delete File",
    description="Delete a specific file from the knowledge base"
)
def delete_specific_file(filename: str):
    """
    Delete a specific file from the knowledge base.
    
    This removes the file and its chunks from vector DB, updates stats,
    and rebuilds the QA chain.
    
    Args:
        filename: Name of the file to delete
        
    Raises:
        InvalidRequest: If file not found
    """
    logger.info(f"Delete request for file: {filename}")
    
    success = doc_store.remove_file(filename)
    
    if not success:
        logger.warning(f"Delete failed - file not found: {filename}")
        raise InvalidRequest(f"File '{filename}' not found in the knowledge base")
    
    remaining_files = len(doc_store.file_metadata)
    logger.info(f"Successfully deleted {filename}. Remaining files: {remaining_files}")
    
    return DeleteFileResponse(
        message=f"File '{filename}' deleted successfully",
        deleted_file=filename,
        remaining_files=remaining_files
    )


@app.get("/export",
    summary="Export Knowledge Base",
    description="Export all documents and metadata for backup"
)
def export_knowledge_base():
    """
    Export the complete knowledge base including all doc content,
    file metadata, and processing stats.
    
    Useful for backup/restore, migration, or data analysis.
    """
    logger.info("Knowledge base export requested")
    
    try:
        exported_data = doc_store.export_data()
        
        logger.info(f"Export successful - {len(exported_data['documents'])} documents")
        
        return {
            "message": "Data exported successfully",
            "export_size": len(exported_data['documents']),
            "data": exported_data
        }
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=DEBUG_MODE)
        raise VectorStoreSyncError("export", str(e))


# Debug endpoint (conditional)

async def verify_debug_enabled():
    """Check if debug endpoints are enabled"""
    if not ENABLE_DEBUG_ENDPOINTS:
        raise HTTPException(
            status_code=404,
            detail="Endpoint not found. Enable ENABLE_DEBUG_ENDPOINTS=true to access debug endpoints."
        )
    return True


@app.get("/debug/qdrant",
    summary="Debug Qdrant Status",
    description="[DEBUG ONLY] Get detailed Qdrant vector store status",
    tags=["Debug"],
    dependencies=[Depends(verify_debug_enabled)]
)
def debug_qdrant():
    """
    Debug endpoint to inspect Qdrant collection status.
    
    Only accessible when ENABLE_DEBUG_ENDPOINTS=true in .env
    
    Returns collection info, point counts, vector store status, in-memory counts
    """
    logger.debug("Debug endpoint accessed: /debug/qdrant")
    
    try:
        if doc_store.qdrant_client:
            collection_info = doc_store.qdrant_client.get_collection(COLLECTION_NAME)
            points_count_result = doc_store.qdrant_client.count(COLLECTION_NAME, exact=True)
            
            return {
                "collection_exists": True,
                "collection_info": collection_info,
                "points_count": {"count": points_count_result.count},
                "vectorstore_exists": doc_store.qdrant_vectorstore is not None,
                "qa_chain_exists": doc_store.qa_chain is not None,
                "in_memory_counts": {
                    "documents": len(doc_store.all_documents),
                    "text_chunks": len(doc_store.all_texts),
                    "files": len(doc_store.file_metadata)
                },
                "reranking_enabled": ENABLE_RERANKING,
                "contextual_enrichment_enabled": ENABLE_CONTEXTUAL_ENRICHMENT
            }
        else:
            return {
                "collection_exists": False,
                "error": "No Qdrant client - using in-memory storage",
                "in_memory_counts": {
                    "documents": len(doc_store.all_documents),
                    "text_chunks": len(doc_store.all_texts),
                    "files": len(doc_store.file_metadata)
                }
            }
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}", exc_info=True)
        return {"error": str(e)}


@app.post("/ingest",
    response_model=IngestResponse,
    summary="Upload Document",
    description="Upload and process a document for the knowledge base"
)
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a document and add it to the knowledge base.
    
    Process: validate file -> parse content -> split into chunks with optional
    enrichment -> generate embeddings -> store in vector DB -> build/rebuild QA chain
    
    Supported formats: PDF, Word (doc/docx), Excel (xls/xlsx), PowerPoint (ppt/pptx),
    Text, Markdown, HTML, CSV, JSON, XML
    
    Args:
        file: Uploaded file (max 100MB)
        
    Returns:
        IngestResponse with processing statistics
        
    Raises:
        FileTooLarge: If file exceeds 100MB
        UnsupportedFileType: If file format not supported
        FileAlreadyExists: If filename already in knowledge base
        FileProcessingError: If processing fails
    """
    logger.info(f"Ingestion started: {file.filename} ({format_file_size(file.size or 0)})")
    
    validate_uploaded_file(file, set(doc_store.file_metadata.keys()))

    temp_file_path = None
    start_time = time.time()
    
    try:
        # create temp directory
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)

        # save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # get file info
        file_size = os.path.getsize(temp_file_path)
        file_type = get_file_type_from_extension(file.filename)
        
        logger.debug(f"File saved - Type: {file_type}, Size: {format_file_size(file_size)}")

        # process document (includes contextual enrichment if enabled)
        loop = asyncio.get_event_loop()
        documents, texts, processing_time = await loop.run_in_executor(
            executor,
            process_document_sync,
            temp_file_path,
            file.filename
        )

        logger.debug(f"Document processed - {len(documents)} docs, {len(texts)} chunks in {processing_time:.2f}s")

        # prepare file metadata
        file_info = {
            'type': file_type,
            'size': file_size,
            'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processing_time': processing_time
        }

        # add to document store
        doc_store.add_documents(documents, texts, file.filename, file_info)

        total_time = time.time() - start_time
        
        logger.info(
            f"Ingestion complete: {file.filename} - "
            f"{len(texts)} chunks created in {total_time:.2f}s total"
        )

        return IngestResponse(
            message=f"File '{file.filename}' ingested successfully and added to persistent knowledge base.",
            num_documents=len(documents),
            num_chunks=len(texts),
            total_files=len(doc_store.file_metadata),
            total_documents=len(doc_store.all_documents),
            total_chunks=len(doc_store.all_texts),
            file_type=file_type,
            file_size_formatted=format_file_size(file_size),
            processing_time=f"{total_time:.2f}s",
            contextual_enrichment_applied=ENABLE_CONTEXTUAL_ENRICHMENT
        )

    except ValueError as ve:
        logger.error(f"Validation error during ingestion of {file.filename}: {str(ve)}")
        raise FileProcessingError(file.filename, str(ve))
    except Exception as e:
        logger.error(f"Error during ingestion of {file.filename}: {str(e)}", exc_info=DEBUG_MODE)
        raise FileProcessingError(file.filename, str(e))
    finally:
        # cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Temporary file cleaned up: {temp_file_path}")


@app.post("/chat",
    summary="Query Knowledge Base",
    description="Ask questions and get AI-generated answers from uploaded documents"
)
async def chat_with_docs(request: ChatRequest):
    """
    Query the RAG system with natural language questions.
    
    Process:
    1. Retrieve relevant chunks (hybrid BM25 + vector search)
    2. Apply cross-encoder reranking (if enabled)
    3. Generate answer using LLM with conversation history
    4. Return answer with source attribution
    
    Uses hybrid retrieval (BM25 + semantic search), cross-encoder reranking for
    relevance filtering, contextually enriched chunks, and conversation history
    for context-aware responses.
    
    Args:
        request: ChatRequest with query string
        
    Returns:
        Dict with response, source_files, source_details, reranking_scores (if enabled),
        and response_time
        
    Raises:
        NoDocumentsIngested: If knowledge base is empty
        RetrievalFailed: If document retrieval fails
        LLMServiceUnavailable: If LLM service is down
        LLMRateLimitExceeded: If API rate limit hit
    """
    start_time = time.time()
    
    logger.info(f"Chat query received: '{request.query[:100]}...'")
    
    # ensure Qdrant client is ready
    doc_store.ensure_client_is_ready()
    
    # rebuild QA chain if needed
    if doc_store.qa_chain is None and doc_store.has_documents():
        logger.debug("Rebuilding QA chain...")
        doc_store._rebuild_qa_chain()
        
    if not doc_store.has_documents():
        logger.warning("Chat attempted with empty knowledge base")
        raise NoDocumentsIngested()
    
    if doc_store.qa_chain is None:
        logger.error("QA chain failed to initialize despite having documents")
        raise RetrievalFailed(request.query, "Knowledge base has documents but QA chain failed to initialize.")

    try:
        # run QA chain in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: doc_store.qa_chain.invoke({
                'question': request.query,
                'chat_history': doc_store.conversation_history[-MAX_CONVERSATION_HISTORY//2:]
            })
        )
        
        response = result['answer']
        source_docs = result.get('source_documents', [])
        
        # process source documents
        source_files = []
        file_types = []
        source_details = []
        
        for doc in source_docs:
            source_file = doc.metadata.get('source_file', 'Unknown')
            file_type = doc.metadata.get('file_type', 'Unknown')
            
            if source_file not in source_files:
                source_files.append(source_file)
            if file_type not in file_types:
                file_types.append(file_type)
            
            # use original content if available (before enrichment)
            content_preview = doc.metadata.get('original_content', doc.page_content)
            source_details.append({
                'file': source_file,
                'type': file_type,
                'chunk_position': doc.metadata.get('chunk_position', 'N/A'),
                'enriched': doc.metadata.get('enriched', False),
                'rerank_score': doc.metadata.get('rerank_score'),
                'rerank_position': doc.metadata.get('rerank_position'),
                'content_preview': content_preview[:200] + '...' if len(content_preview) > 200 else content_preview
            })

        # log retrieval results (debug mode only)
        if DEBUG_MODE:
            logger.debug(f"Retrieved {len(source_docs)} chunks from {len(source_files)} files: {source_files}")
            if ENABLE_RERANKING and source_docs:
                top_score = source_docs[0].metadata.get('rerank_score', 'N/A')
                logger.debug(f"Reranking applied - Top score: {top_score}")

        # update conversation history
        doc_store.conversation_history.append((request.query, response))
        if len(doc_store.conversation_history) > MAX_CONVERSATION_HISTORY:
            doc_store.conversation_history = doc_store.conversation_history[-MAX_CONVERSATION_HISTORY:]

        processing_time = time.time() - start_time
        
        logger.info(f"Chat response generated in {processing_time:.2f}s - {len(source_docs)} sources used")

        return {
            "response": response,
            "source_files": source_files,
            "file_types": file_types,
            "num_sources": len(source_docs),
            "source_details": source_details,
            "total_files_in_knowledge_base": len(doc_store.file_metadata),
            "response_time": f"{processing_time:.2f}s",
            "storage_type": "persistent" if doc_store.qdrant_client else "in-memory",
            "reranking_enabled": ENABLE_RERANKING,
            "contextual_enrichment_enabled": ENABLE_CONTEXTUAL_ENRICHMENT,
            "reranking_scores": [
                {
                    'file': doc.metadata.get('source_file'),
                    'score': doc.metadata.get('rerank_score'),
                    'position': doc.metadata.get('rerank_position')
                }
                for doc in source_docs if 'rerank_score' in doc.metadata
            ] if ENABLE_RERANKING else []
        }
        
    except Exception as e:
        logger.error(f"Error during chat processing: {str(e)}", exc_info=DEBUG_MODE)
        
        # identify specific error types
        error_str = str(e).lower()
       
        if "rate limit" in error_str or "429" in error_str:
            logger.warning("Rate limit exceeded")
            raise LLMRateLimitExceeded()
        elif "timeout" in error_str:
            logger.warning("LLM request timeout")
            raise LLMTimeout()
        elif "connection" in error_str or "groq" in error_str:
            logger.error("LLM service connection failed")
            raise LLMServiceUnavailable("Groq", str(e))
        else:
            logger.error(f"Retrieval failed: {str(e)}")
            raise RetrievalFailed(request.query, str(e))



# App lifecycle events

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("="*60)
    logger.info(f"{APP_TITLE} v{APP_VERSION}")
    logger.info("="*60)
    logger.info(f"LLM Provider: {LLM_PROVIDER}")
    logger.info(f"Reranking: {'Enabled' if ENABLE_RERANKING else 'Disabled'}")
    logger.info(f"Contextual Enrichment: {'Enabled' if ENABLE_CONTEXTUAL_ENRICHMENT else 'Disabled'}")
    logger.info(f"Debug Mode: {DEBUG_MODE}")
    logger.info(f"Debug Endpoints: {ENABLE_DEBUG_ENDPOINTS}")
    logger.info(f"Storage: {doc_store.get_stats()['storage_type']}")
    logger.info("="*60)


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("Application shutdown initiated")
    executor.shutdown(wait=True)
    logger.info("Thread pool executor shut down")
    logger.info("Application shutdown complete")