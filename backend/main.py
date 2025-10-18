import os
import asyncio
from typing import Any, List
import time
from datetime import datetime

from exceptions import *
from config import *
from utils import *
from services import DocumentStore, process_document_sync

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

# === NEW: Register Custom Error Handlers ===
register_error_handlers(app) 

# Enable CORS to allow the frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

@app.get("/")
def read_root():
    return {
        "message": APP_TITLE,
        "version": APP_VERSION,
        "features": APP_FEATURES,
        "storage": get_storage_info(),
        "llm_provider": LLM_PROVIDER,
        "reranking_enabled": ENABLE_RERANKING,
        "contextual_enrichment_enabled": ENABLE_CONTEXTUAL_ENRICHMENT
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    stats = doc_store.get_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "reranking_enabled": ENABLE_RERANKING,
        "contextual_enrichment_enabled": ENABLE_CONTEXTUAL_ENRICHMENT
    }

@app.get("/stats", response_model=SystemStatsResponse)
def get_system_stats():
    """Get detailed system statistics"""
    stats = doc_store.get_stats()
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

@app.get("/files", response_model=FileListResponse)
def get_uploaded_files():
    """Get list of all uploaded files with complete metadata"""
    files = doc_store.get_file_list()
    return FileListResponse(
        files=files,
        total_files=len(files),
        total_documents=len(doc_store.all_documents),
        total_chunks=len(doc_store.all_texts)
    )

@app.delete("/files")
def clear_all_files():
    """Clear all uploaded files and reset the system"""
    doc_store.clear_all()
    return {"message": "All files cleared successfully"}

@app.delete("/files/{filename}", response_model=DeleteFileResponse)
def delete_specific_file(filename: str):
    """Delete a specific file from the knowledge base"""
    success = doc_store.remove_file(filename)
    
    if not success:
        raise InvalidRequest(f"File '{filename}' not found in the knowledge base")
    
    remaining_files = len(doc_store.file_metadata)
    return DeleteFileResponse(
        message=f"File '{filename}' deleted successfully",
        deleted_file=filename,
        remaining_files=remaining_files
    )

@app.get("/export")
def export_knowledge_base():
    """Export all documents and metadata for backup"""
    try:
        exported_data = doc_store.export_data()
        return {
            "message": "Data exported successfully",
            "export_size": len(exported_data['documents']),
            "data": exported_data
        }
    except Exception as e:
        raise VectorStoreSyncError("export", str(e))

@app.get("/debug/qdrant")
def debug_qdrant():
    """Debug Qdrant collection status - useful for troubleshooting persistence"""
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
        return {"error": str(e)}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingests a file and adds it to the existing vector store with enhanced performance and persistence.
    Now includes contextual enrichment.
    """
    print(f"Received file for ingestion: {file.filename} ({format_file_size(file.size or 0)})")
    
    validate_uploaded_file(file, set(doc_store.file_metadata.keys()))

    temp_file_path = None
    start_time = time.time()
    
    try:
        # Create a temporary directory if it doesn't exist
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)

        # Save the uploaded content to the temporary file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get file information
        file_size = os.path.getsize(temp_file_path)
        file_type = get_file_type_from_extension(file.filename)
        
        print(f"File details - Type: {file_type}, Size: {format_file_size(file_size)}")

        # Process document in thread pool (includes contextual enrichment)
        loop = asyncio.get_event_loop()
        documents, texts, processing_time = await loop.run_in_executor(
            executor, 
            process_document_sync, 
            temp_file_path, 
            file.filename
        )

        # Prepare file info for storage
        file_info = {
            'type': file_type,
            'size': file_size,
            'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processing_time': processing_time
        }

        # Add documents to the store
        doc_store.add_documents(documents, texts, file.filename, file_info)

        total_time = time.time() - start_time
        print(f"Total ingestion time for {file.filename}: {total_time:.2f}s")

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
        print(f"Validation error during ingestion: {ve}")
        raise FileProcessingError(file.filename, str(ve))
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise FileProcessingError(file.filename, str(e))
    finally:
        # Cleanup the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat")
async def chat_with_docs(request: ChatRequest):
    """
    Answers a user's query based on all ingested documents.
    Now includes reranking for improved relevance.
    """
    start_time = time.time()
    
    # Ensure client is ready
    doc_store.ensure_client_is_ready() 
    
    # Rebuild QA chain if needed (this also correctly sets it to None if KB is empty)
    if doc_store.qa_chain is None and doc_store.has_documents():
        doc_store._rebuild_qa_chain()
        
    if not doc_store.has_documents():
        # Even if qa_chain is None, this is the clearer and faster check for an empty KB
        raise NoDocumentsIngested()
    
    if doc_store.qa_chain is None:
        # Fallback if has_documents is True but qa_chain failed to build
        raise RetrievalFailed(request.query, "Knowledge base has documents but QA chain failed to initialize.")

    try:
        # Run the QA chain in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: doc_store.qa_chain.invoke({
                'question': request.query,
                'chat_history': doc_store.conversation_history[-MAX_CONVERSATION_HISTORY//2:]
            })
        )
        
        response = result['answer']
        
        # Get source documents with detailed metadata
        source_docs = result.get('source_documents', [])
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
            
            # Add detailed source info for debugging
            # Use original content if available (before enrichment)
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

        # Log retrieved sources for debugging
        print(f"\n=== RETRIEVAL DEBUG ===")
        print(f"Query: {request.query}")
        print(f"Retrieved {len(source_docs)} chunks from {len(source_files)} files: {source_files}")
        if ENABLE_RERANKING and source_docs:
            print(f"Reranking applied: Top score = {source_docs[0].metadata.get('rerank_score', 'N/A')}")
        print(f"======================\n")

        # Update conversation history
        doc_store.conversation_history.append((request.query, response))
        if len(doc_store.conversation_history) > MAX_CONVERSATION_HISTORY:
            doc_store.conversation_history = doc_store.conversation_history[-MAX_CONVERSATION_HISTORY:]

        processing_time = time.time() - start_time
        print(f"Chat response generated in {processing_time:.2f}s")

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
        print(f"Error during chat: {e}")
        # Try to identify specific error types
        error_str = str(e).lower() 
       
        if "rate limit" in error_str or "429" in error_str:
            raise LLMRateLimitExceeded()
        elif "timeout" in error_str:
            raise LLMTimeout()
        elif "connection" in error_str or "groq" in error_str:
            raise LLMServiceUnavailable("Groq", str(e))
        else:
            # Note: This is now a true retrieval failure (e.g., Qdrant is down)
            # as the empty KB case is handled above.
            raise RetrievalFailed(request.query, str(e))

# Cleanup function for graceful shutdown
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup resources on shutdown"""
    executor.shutdown(wait=True)
    print("FastAPI application shutdown complete")