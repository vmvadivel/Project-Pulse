import os
import shutil
import asyncio
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import time
import threading
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Check for the GROQ API key
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it.")

# Initialize Groq Langchain chat model
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Multi-File RAG System",
    description="High-performance RAG system with concurrent user support",
    version="2.0.0"
)

# Enable CORS to allow the frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# File type detection helper
def get_file_type_from_extension(filename: str) -> str:
    """Get file type from file extension."""
    ext = os.path.splitext(filename)[1].lower()
    file_type_map = {
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
    return file_type_map.get(ext, 'Unknown')

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def process_document_sync(temp_file_path: str, filename: str) -> tuple:
    """
    Synchronous document processing function to run in thread pool.
    Returns: (documents, texts, processing_time)
    """
    start_time = time.time()
    
    # Determine the loader based on file extension
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        loader = UnstructuredPDFLoader(temp_file_path)
    else:
        loader = UnstructuredFileLoader(temp_file_path)

    print(f"Starting document loading for {filename}...")
    documents = loader.load()
    print(f"Document loading complete for {filename}. Loaded {len(documents)} documents.")

    # Filter out empty documents
    documents = [doc for doc in documents if doc.page_content.strip()]
    
    if not documents:
        raise ValueError(f"No readable content found in {filename}. The file might be empty or corrupted.")

    # Using RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Increased for better context
        chunk_overlap=100,  # Increased overlap
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    print(f"Starting text chunking for {filename}...")
    texts = text_splitter.split_documents(documents)
    print(f"Text chunking complete for {filename}. Created {len(texts)} text chunks.")

    # Filter out very short chunks
    texts = [text for text in texts if len(text.page_content.strip()) > 20]
    
    if not texts:
        raise ValueError(f"No meaningful text chunks created from {filename}.")

    processing_time = time.time() - start_time
    return documents, texts, processing_time

# Enhanced in-memory storage for multiple files with thread safety
class DocumentStore:
    def __init__(self):
        self.all_documents = []  # Store all documents from all files
        self.all_texts = []      # Store all text chunks from all files
        self.file_metadata = {}  # Store metadata about each file
        self.qdrant_vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        self._lock = threading.RLock()  # Thread safety
    
    def add_documents(self, documents, texts, filename, file_info):
        """Add new documents and texts to the store with thread safety"""
        with self._lock:
            # Add metadata to track which file each document comes from
            for doc in documents:
                doc.metadata.update({
                    'source_file': filename,
                    'file_type': file_info['type'],
                    'file_size': file_info['size']
                })
            for text in texts:
                text.metadata.update({
                    'source_file': filename,
                    'file_type': file_info['type'],
                    'file_size': file_info['size']
                })
                
            self.all_documents.extend(documents)
            self.all_texts.extend(texts)
            
            # Store enhanced file metadata
            self.file_metadata[filename] = {
                'num_documents': len(documents),
                'num_chunks': len(texts),
                'file_type': file_info['type'],
                'file_size': file_info['size'],
                'file_size_formatted': format_file_size(file_info['size']),
                'upload_date': file_info.get('upload_date', 'Unknown'),
                'processing_time': file_info.get('processing_time', 0)
            }
            
            # Rebuild the vector store and QA chain with all documents
            self._rebuild_qa_chain()
    
    def remove_file(self, filename: str) -> bool:
        """Remove a specific file and its associated documents"""
        with self._lock:
            if filename not in self.file_metadata:
                return False
            
            # Remove documents and texts associated with this file
            self.all_documents = [doc for doc in self.all_documents 
                                if doc.metadata.get('source_file') != filename]
            self.all_texts = [text for text in self.all_texts 
                            if text.metadata.get('source_file') != filename]
            
            # Remove file metadata
            del self.file_metadata[filename]
            
            # Rebuild the vector store and QA chain
            self._rebuild_qa_chain()
            return True
    
    def _rebuild_qa_chain(self):
        """Rebuild the QA chain with all documents"""
        if not self.all_texts:
            self.qdrant_vectorstore = None
            self.qa_chain = None
            return
                
        # Create/update Qdrant vector store with all texts
        self.qdrant_vectorstore = Qdrant.from_documents(
            documents=self.all_texts,
            embedding=embeddings,
            location=":memory:",
            collection_name="all_documents"
        )
        
        # Create retrievers with optimized settings for performance
        qdrant_retriever = self.qdrant_vectorstore.as_retriever(
            search_kwargs={'k': 5, 'fetch_k': 10}  # Fetch more candidates for better results
        )
        
        # Create BM25 retriever from all texts
        bm25_retriever = BM25Retriever.from_documents(documents=self.all_texts)
        bm25_retriever.k = 5
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever],
            weights=[0.3, 0.7]  # Favor vector search slightly
        )
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=ensemble_retriever,
            return_source_documents=True
        )
    
    def get_file_list(self):
        """Get list of uploaded files with metadata"""
        with self._lock:
            return [
                {
                    'filename': filename,
                    'num_documents': metadata['num_documents'],
                    'num_chunks': metadata['num_chunks'],
                    'file_type': metadata['file_type'],
                    'file_size': metadata['file_size'],
                    'file_size_formatted': metadata['file_size_formatted'],
                    'upload_date': metadata.get('upload_date', 'Unknown'),
                    'processing_time': f"{metadata.get('processing_time', 0):.2f}s"
                }
                for filename, metadata in self.file_metadata.items()
            ]
    
    def clear_all(self):
        """Clear all stored documents"""
        with self._lock:
            self.all_documents = []
            self.all_texts = []
            self.file_metadata = {}
            self.qdrant_vectorstore = None
            self.qa_chain = None
            self.conversation_history = []

    def get_stats(self):
        """Get system statistics"""
        with self._lock:
            return {
                'total_files': len(self.file_metadata),
                'total_documents': len(self.all_documents),
                'total_chunks': len(self.all_texts),
                'total_size': sum(meta['file_size'] for meta in self.file_metadata.values()),
                'avg_processing_time': sum(meta.get('processing_time', 0) for meta in self.file_metadata.values()) / max(len(self.file_metadata), 1)
            }

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

@app.get("/")
def read_root():
    return {
        "message": "Enhanced Multi-File RAG System with Performance Optimizations",
        "version": "2.0.0",
        "features": ["Concurrent Processing", "Single File Delete", "Performance Monitoring"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    stats = doc_store.get_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "stats": stats
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
        avg_processing_time=f"{stats['avg_processing_time']:.2f}s"
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
        raise HTTPException(
            status_code=404, 
            detail=f"File '{filename}' not found in the knowledge base"
        )
    
    remaining_files = len(doc_store.file_metadata)
    return DeleteFileResponse(
        message=f"File '{filename}' deleted successfully",
        deleted_file=filename,
        remaining_files=remaining_files
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingests a file and adds it to the existing vector store with enhanced performance.
    Uses async processing to handle large files without blocking the event loop.
    """
    print(f"Received file for ingestion: {file.filename} ({format_file_size(file.size or 0)})")
    
    # Validate file size (100MB limit)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size allowed is {format_file_size(MAX_FILE_SIZE)}"
        )
    
    # Check if file already exists
    if file.filename in doc_store.file_metadata:
        raise HTTPException(
            status_code=400, 
            detail=f"File '{file.filename}' has already been uploaded. Please use a different name or delete the existing file first."
        )
    
    temp_file_path = None
    start_time = time.time()
    
    try:
        # Create a temporary directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        temp_file_path = os.path.join("temp", file.filename)

        # Save the uploaded content to the temporary file (async)
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get file information
        file_size = os.path.getsize(temp_file_path)
        file_type = get_file_type_from_extension(file.filename)
        
        print(f"File details - Type: {file_type}, Size: {format_file_size(file_size)}")

        # Process document in thread pool to avoid blocking event loop
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

        # Add documents to the store (this also rebuilds the QA chain)
        doc_store.add_documents(documents, texts, file.filename, file_info)

        total_time = time.time() - start_time
        print(f"Total ingestion time for {file.filename}: {total_time:.2f}s")

        return IngestResponse(
            message=f"File '{file.filename}' ingested successfully and added to existing knowledge base.",
            num_documents=len(documents),
            num_chunks=len(texts),
            total_files=len(doc_store.file_metadata),
            total_documents=len(doc_store.all_documents),
            total_chunks=len(doc_store.all_texts),
            file_type=file_type,
            file_size_formatted=format_file_size(file_size),
            processing_time=f"{total_time:.2f}s"
        )

    except ValueError as ve:
        print(f"Validation error during ingestion: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")
    finally:
        # Cleanup the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat")
async def chat_with_docs(request: ChatRequest):
    """
    Answers a user's query based on all ingested documents with enhanced source information.
    Optimized for concurrent user support.
    """
    start_time = time.time()
    
    if doc_store.qa_chain is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents have been ingested yet. Please upload files first using the /ingest endpoint."
        )

    try:
        # Run the QA chain in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: doc_store.qa_chain.invoke({
                'question': request.query,
                'chat_history': doc_store.conversation_history[-10:]  # Limit history for performance
            })
        )
        
        response = result['answer']
        
        # Get source documents to show which files contributed to the answer
        source_docs = result.get('source_documents', [])
        source_files = []
        file_types = []
        
        for doc in source_docs:
            source_file = doc.metadata.get('source_file', 'Unknown')
            file_type = doc.metadata.get('file_type', 'Unknown')
            
            if source_file not in source_files:
                source_files.append(source_file)
            if file_type not in file_types:
                file_types.append(file_type)

        # Update conversation history (keep last 20 exchanges)
        doc_store.conversation_history.append((request.query, response))
        if len(doc_store.conversation_history) > 20:
            doc_store.conversation_history = doc_store.conversation_history[-20:]

        processing_time = time.time() - start_time
        print(f"Chat response generated in {processing_time:.2f}s")

        return {
            "response": response,
            "source_files": source_files,
            "file_types": file_types,
            "num_sources": len(source_docs),
            "total_files_in_knowledge_base": len(doc_store.file_metadata),
            "response_time": f"{processing_time:.2f}s"
        }
        
    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")

# Cleanup function for graceful shutdown
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup resources on shutdown"""
    executor.shutdown(wait=True)
    print("FastAPI application shutdown complete")