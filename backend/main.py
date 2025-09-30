import os
import shutil
import asyncio
import uuid
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from fastapi.middleware.cors import CORSMiddleware
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import time
import threading
from datetime import datetime
import json
import gc

# Load environment variables from .env file
load_dotenv()

# Check for the GROQ API key
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it.")

# Configuration
QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "./qdrant_storage")
COLLECTION_NAME = "all_documents"

# Initialize Groq Langchain chat model
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Multi-File RAG System with Persistence",
    description="High-performance RAG system with persistent storage and concurrent user support",
    version="2.1.0"
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

def initialize_qdrant_client():
    """Initialize Qdrant client with persistent storage."""
    try:
        # Ensure storage directory exists
        os.makedirs(QDRANT_STORAGE_PATH, exist_ok=True)
        
        # Create Qdrant client with persistent storage
        client = QdrantClient(path=QDRANT_STORAGE_PATH)
        
        print(f"Qdrant client initialized with storage at: {QDRANT_STORAGE_PATH}")
        return client
    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        raise

def ensure_collection_exists(client: QdrantClient, collection_name: str, vector_size: int = 768):
    """Ensure the collection exists, create if it doesn't."""
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            print(f"Creating new collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created successfully")
        else:
            print(f"Collection '{collection_name}' already exists")
            
    except Exception as e:
        print(f"Error managing collection: {e}")
        raise

def save_metadata_to_file(file_metadata: dict):
    """Save file metadata to persistent storage."""
    try:
        metadata_path = os.path.join(QDRANT_STORAGE_PATH, "file_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(file_metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving metadata: {e}")

def load_metadata_from_file() -> dict:
    """Load file metadata from persistent storage."""
    try:
        metadata_path = os.path.join(QDRANT_STORAGE_PATH, "file_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
    return {}

def serialize_metadata(metadata: dict) -> dict:
    """Handle complex metadata objects for JSON serialization."""
    serialized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            serialized[key] = value
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = str(value)
    return serialized

def deserialize_metadata(metadata: dict) -> dict:
    """Reconstruct metadata from JSON."""
    return metadata

def serialize_documents(documents: List[Document]) -> List[dict]:
    """Convert LangChain documents to JSON-serializable format."""
    return [
        {
            'page_content': doc.page_content,
            'metadata': serialize_metadata(doc.metadata)
        } 
        for doc in documents
    ]

def deserialize_documents(doc_data: List[dict]) -> List[Document]:
    """Convert JSON data back to LangChain documents."""
    return [
        Document(
            page_content=item['page_content'],
            metadata=deserialize_metadata(item['metadata'])
        )
        for item in doc_data
    ]

def save_documents_to_json(documents: List[Document], file_path: str):
    """Safely save documents to JSON file."""
    try:
        serialized = serialize_documents(documents)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving documents to {file_path}: {e}")
        raise

def load_documents_from_json(file_path: str) -> List[Document]:
    """Safely load documents from JSON file."""
    try:
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        return deserialize_documents(doc_data)
    except Exception as e:
        print(f"Error loading documents from {file_path}: {e}")
        return []

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
        chunk_size=1000,  # Increased to capture more complete booking info
        chunk_overlap=200,  # Increased overlap to avoid splitting related info
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

# Enhanced in-memory storage for multiple files with thread safety and persistence
class DocumentStore:
    def __init__(self):
        self.all_documents = []
        self.all_texts = []
        self.file_metadata = {}
        self.qdrant_client = None
        self.qdrant_vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        self._lock = threading.RLock()
        
        # Initialize persistent storage
        self._initialize_persistent_storage()
    
    def _initialize_persistent_storage(self):
        """Initialize Qdrant with persistent storage and load existing data."""
        try:
            # Initialize Qdrant client with persistent storage
            self.qdrant_client = initialize_qdrant_client()
            
            # Ensure collection exists
            ensure_collection_exists(self.qdrant_client, COLLECTION_NAME)
            
            # Load existing complete state
            self._load_complete_state()
            
            # If we have existing data, rebuild the QA chain
            if self.file_metadata:
                print(f"Found {len(self.file_metadata)} files in persistent storage")
                print(f"Loaded {len(self.all_documents)} documents and {len(self.all_texts)} text chunks")
                self._rebuild_qa_chain()
            else:
                print("No existing data found, starting fresh")
                
        except Exception as e:
            print(f"Error initializing persistent storage: {e}")
            print("Falling back to in-memory storage")
            self.qdrant_client = None

    def ensure_client_is_ready(self):
        """Checks if the Qdrant client is None and re-initializes it if needed."""
        if self.qdrant_client is None:
            print("[RE-INIT] Client is None. Attempting to re-initialize Qdrant client and collection...")
            try:
                self.qdrant_client = initialize_qdrant_client()
                ensure_collection_exists(self.qdrant_client, COLLECTION_NAME)
                print("[RE-INIT] Qdrant client and collection successfully re-initialized.")
            except Exception as e:
                print(f"[RE-INIT] Error re-initializing Qdrant client: {e}")
                self.qdrant_client = None
    
    def _save_complete_state(self):
        """Save all documents, texts, and metadata to JSON files."""
        try:
            base_path = QDRANT_STORAGE_PATH
            os.makedirs(base_path, exist_ok=True)
            
            # Save documents
            docs_path = os.path.join(base_path, "documents.json")
            save_documents_to_json(self.all_documents, docs_path)
            
            # Save text chunks  
            texts_path = os.path.join(base_path, "text_chunks.json")
            save_documents_to_json(self.all_texts, texts_path)
            
            # Save metadata
            save_metadata_to_file(self.file_metadata)
            
            print(f"Saved complete state: {len(self.all_documents)} documents, {len(self.all_texts)} text chunks")
            
        except Exception as e:
            print(f"Error saving complete state: {e}")
    
    def _load_complete_state(self):
        """Load all documents, texts, and metadata from JSON files."""
        try:
            base_path = QDRANT_STORAGE_PATH
            
            # Load documents
            docs_path = os.path.join(base_path, "documents.json")
            self.all_documents = load_documents_from_json(docs_path)
            
            # Load text chunks
            texts_path = os.path.join(base_path, "text_chunks.json")
            self.all_texts = load_documents_from_json(texts_path)
            
            # Load metadata
            self.file_metadata = load_metadata_from_file()
            
        except Exception as e:
            print(f"Error loading complete state: {e}")
            self.all_documents = []
            self.all_texts = []
            self.file_metadata = {}
    
    def add_documents(self, documents, texts, filename, file_info):
        """Add new documents and texts to the store with thread safety and persistence"""
        with self._lock:
            # Ensure client is ready before attempting ingestion
            self.ensure_client_is_ready() 
            
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
            
            if self.qdrant_client:
                print(f"[DEBUG] Starting Qdrant upsert process for {len(texts)} chunks...")
                
                try:
                    # Generate vectors
                    text_contents = [text.page_content for text in texts]
                    vectors = embeddings.embed_documents(text_contents)
                    
                    if not vectors or len(vectors) != len(texts):
                        raise Exception("Vector generation failed or vector count mismatch.")

                    print(f"[DEBUG] Vectorization complete. Created {len(vectors)} vectors.")

                    # Create PointStructs for Qdrant
                    points = [
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload=text.metadata
                        )
                        for vector, text in zip(vectors, texts)
                    ]

                    # Upsert the points
                    print(f"[DEBUG] Calling qdrant_client.upsert for {len(points)} points...")
                    self.qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                        wait=True
                    )
                    print(f"Successfully upserted {len(points)} points to Qdrant.")
                    
                    # Verify count
                    final_count_result = self.qdrant_client.count(COLLECTION_NAME, exact=True)
                    print(f"--- QDRANT PERSISTENCE CHECK ---")
                    print(f"Upsert Complete. Current Qdrant Count: {final_count_result.count}")
                    print(f"--------------------------------")
                    
                except Exception as e:
                    print(f"!!! CRITICAL QDRANT ERROR during upsert: {e}")
            
            # Save to persistent storage (JSON)
            self._save_complete_state()
            
            # Rebuild the vector store and QA chain
            self._rebuild_qa_chain()
    
    def remove_file(self, filename: str) -> bool:
        """Remove a specific file and its associated documents"""
        with self._lock:
            if filename not in self.file_metadata:
                return False
            
            self.ensure_client_is_ready() 

            try:
                # Remove from Qdrant
                if self.qdrant_client:
                    self.qdrant_client.delete(
                        collection_name=COLLECTION_NAME,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="source_file", 
                                    match=MatchValue(value=filename)
                                )
                            ]
                        ),
                        wait=True
                    )
                
                # Remove documents and texts
                self.all_documents = [doc for doc in self.all_documents 
                                    if doc.metadata.get('source_file') != filename]
                self.all_texts = [text for text in self.all_texts 
                                if text.metadata.get('source_file') != filename]
                
                # Remove file metadata
                del self.file_metadata[filename]
                
                # Save updated state
                self._save_complete_state()
                
                # Rebuild the QA chain
                self._rebuild_qa_chain()
                return True
                
            except Exception as e:
                print(f"Error removing file {filename}: {e}")
                return False
    
    def _create_qa_prompt(self):
        """Create an enhanced prompt for better multi-document retrieval"""
        from langchain.prompts import PromptTemplate
        
        template = """You are a helpful assistant answering questions based on the provided context from documents.

CRITICAL INSTRUCTIONS:
1. Read through ALL the context provided below carefully - it may come from multiple different documents
2. When a question mentions multiple items, entities, or topics, search through the ENTIRE context for each one
3. Synthesize information from different parts of the context when answering
4. If you find information about some aspects of the question but not others, clearly state what you found and what is missing
5. Base your answer ONLY on the provided context - do not make assumptions or add external knowledge
6. If the context does not contain enough information to fully answer the question, say so explicitly

Context:
{context}

Question: {question}

Answer:"""
        
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def _rebuild_qa_chain(self):
        """Rebuild the QA chain with all documents"""
        try:
            if self.qdrant_client:
                self.qdrant_vectorstore = Qdrant(
                    client=self.qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embeddings=embeddings
                )
            else:
                self.qdrant_vectorstore = None

            if not self.all_texts and not self.qdrant_vectorstore:
                self.qa_chain = None
                return
            
            if self.all_texts and self.qdrant_vectorstore:
                qdrant_retriever = self.qdrant_vectorstore.as_retriever(
                    search_kwargs={'k': 10}  # Increased from 5 to retrieve more context
                )
                
                bm25_retriever = BM25Retriever.from_documents(documents=self.all_texts)
                bm25_retriever.k = 10  # Increased from 5
                
                retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, qdrant_retriever],
                    weights=[0.4, 0.6]  # Slightly favor BM25 for keyword matching
                )
            
            elif self.qdrant_vectorstore:
                retriever = self.qdrant_vectorstore.as_retriever(
                    search_kwargs={'k': 10}  # Increased from 5
                )
            else:
                self.qa_chain = None
                return
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={
                    "prompt": self._create_qa_prompt()
                }
            )
            
        except Exception as e:
            print(f"Error rebuilding QA chain: {e}")
            self.qa_chain = None
    
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
        """Clear all stored documents and completely reset persistent storage."""
        with self._lock:
            try:
                # 1. Clear in-memory data first
                self.all_documents = []
                self.all_texts = []
                self.file_metadata = {}
                self.conversation_history = []
                self.qa_chain = None
                
                # 2. Complete Persistent Storage Reset
                if self.qdrant_client:
                    print(f"!!! CRITICAL: Resetting persistent Qdrant/JSON storage at {QDRANT_STORAGE_PATH}...")
                    
                    try:
                        # --- STEP 1: Delete the Vectorstore to release its connection ---
                        if self.qdrant_vectorstore:
                            del self.qdrant_vectorstore
                            self.qdrant_vectorstore = None
                        
                        # --- STEP 2: Delete the Client and force garbage collection ---
                        del self.qdrant_client 
                        self.qdrant_client = None
                        
                        # Force garbage collection to release file locks immediately
                        gc.collect()
                        
                        # Small delay to ensure OS releases locks (Windows can be slow)
                        time.sleep(0.5)
                        
                        # --- STEP 3: Delete the storage directory ---
                        if os.path.exists(QDRANT_STORAGE_PATH):
                            try:
                                shutil.rmtree(QDRANT_STORAGE_PATH)
                                print("Persistent storage successfully deleted.")
                            except PermissionError as pe:
                                # If still locked, try deleting individual files
                                print(f"Permission error deleting folder, attempting file-by-file deletion: {pe}")
                                for root, dirs, files in os.walk(QDRANT_STORAGE_PATH, topdown=False):
                                    for name in files:
                                        try:
                                            os.remove(os.path.join(root, name))
                                        except Exception as e:
                                            print(f"Could not delete file {name}: {e}")
                                    for name in dirs:
                                        try:
                                            os.rmdir(os.path.join(root, name))
                                        except Exception as e:
                                            print(f"Could not delete directory {name}: {e}")
                                # Try one more time to remove the root
                                try:
                                    shutil.rmtree(QDRANT_STORAGE_PATH)
                                except:
                                    print("Warning: Some files may remain locked. They will be overwritten on next initialization.")
                        
                        # --- STEP 4: Immediately re-initialize with fresh storage ---
                        print("Re-initializing fresh Qdrant client and collection...")
                        self._initialize_persistent_storage()
                        print("Fresh Qdrant storage initialized successfully.")
                        
                    except Exception as e:
                        print(f"Error during storage reset: {e}")
                        # Ensure client is None so it can be re-initialized
                        self.qdrant_client = None
                        self.qdrant_vectorstore = None
                        # Try to initialize fresh storage anyway
                        try:
                            self._initialize_persistent_storage()
                        except Exception as init_error:
                            print(f"Failed to re-initialize storage: {init_error}")
                
            except Exception as e:
                print(f"Error clearing all data: {e}")

    def get_stats(self):
        """Get system statistics"""
        with self._lock:
            storage_info = {
                'storage_type': 'persistent' if self.qdrant_client else 'in-memory',
                'storage_path': QDRANT_STORAGE_PATH if self.qdrant_client else 'memory'
            }
            
            return {
                'total_files': len(self.file_metadata),
                'total_documents': len(self.all_documents),
                'total_chunks': len(self.all_texts),
                'total_size': sum(meta['file_size'] for meta in self.file_metadata.values()),
                'avg_processing_time': sum(meta.get('processing_time', 0) for meta in self.file_metadata.values()) / max(len(self.file_metadata), 1),
                **storage_info
            }
    
    def export_data(self) -> dict:
        """Export all documents and metadata for backup"""
        with self._lock:
            return {
                'metadata': self.file_metadata,
                'documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    } for doc in self.all_documents
                ],
                'export_timestamp': datetime.now().isoformat()
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
    storage_type: str
    storage_path: str

@app.get("/")
def read_root():
    return {
        "message": "Enhanced Multi-File RAG System with Persistent Storage",
        "version": "2.1.0",
        "features": ["Persistent Storage", "Concurrent Processing", "Single File Delete", "Performance Monitoring"],
        "storage": {
            "type": "persistent" if doc_store.qdrant_client else "in-memory",
            "path": QDRANT_STORAGE_PATH if doc_store.qdrant_client else "memory"
        }
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
        avg_processing_time=f"{stats['avg_processing_time']:.2f}s",
        storage_type=stats['storage_type'],
        storage_path=stats['storage_path']
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
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

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
                }
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
    """
    print(f"Received file for ingestion: {file.filename} ({format_file_size(file.size or 0)})")
    
    # Validate file size (100MB limit)
    MAX_FILE_SIZE = 100 * 1024 * 1024
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

        # Save the uploaded content to the temporary file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get file information
        file_size = os.path.getsize(temp_file_path)
        file_type = get_file_type_from_extension(file.filename)
        
        print(f"File details - Type: {file_type}, Size: {format_file_size(file_size)}")

        # Process document in thread pool
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
    Answers a user's query based on all ingested documents.
    """
    start_time = time.time()
    
    # Ensure client is ready
    doc_store.ensure_client_is_ready() 
    
    # Rebuild QA chain if needed
    if doc_store.qa_chain is None and doc_store.qdrant_client:
        doc_store._rebuild_qa_chain()
        
    if doc_store.qa_chain is None:
        raise HTTPException(
            status_code=400, 
            detail="No documents have been ingested yet. Please upload files first using the /ingest endpoint."
        )

    try:
        # Run the QA chain in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: doc_store.qa_chain.invoke({
                'question': request.query,
                'chat_history': doc_store.conversation_history[-10:]
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
            source_details.append({
                'file': source_file,
                'type': file_type,
                'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })

        # Log retrieved sources for debugging
        print(f"\n=== RETRIEVAL DEBUG ===")
        print(f"Query: {request.query}")
        print(f"Retrieved {len(source_docs)} chunks from {len(source_files)} files: {source_files}")
        print(f"======================\n")

        # Update conversation history
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
            "source_details": source_details,  # Added for debugging
            "total_files_in_knowledge_base": len(doc_store.file_metadata),
            "response_time": f"{processing_time:.2f}s",
            "storage_type": "persistent" if doc_store.qdrant_client else "in-memory"
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