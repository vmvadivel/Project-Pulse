import os
import shutil
from typing import Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
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
# New Imports for Hybrid Search
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for the GROQ API key
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it.")

# Initialize Groq Langchain chat model
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize FastAPI
app = FastAPI()

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

# Enhanced in-memory storage for multiple files
class DocumentStore:
    def __init__(self):
        self.all_documents = []  # Store all documents from all files
        self.all_texts = []      # Store all text chunks from all files
        self.file_metadata = {}  # Store metadata about each file
        self.qdrant_vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
    
    def add_documents(self, documents, texts, filename, file_info):
        """Add new documents and texts to the store"""
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
            'upload_date': file_info.get('upload_date', 'Unknown')
        }
        
        # Rebuild the vector store and QA chain with all documents
        self._rebuild_qa_chain()
    
    def _rebuild_qa_chain(self):
        """Rebuild the QA chain with all documents"""
        if not self.all_texts:
            return
            
        # Create/update Qdrant vector store with all texts
        self.qdrant_vectorstore = Qdrant.from_documents(
            documents=self.all_texts,
            embedding=embeddings,
            location=":memory:",
            collection_name="all_documents"
        )
        
        # Create retrievers
        qdrant_retriever = self.qdrant_vectorstore.as_retriever(search_kwargs={'k': 10})
        
        # Create BM25 retriever from all texts
        bm25_retriever = BM25Retriever.from_documents(documents=self.all_texts)
        bm25_retriever.k = 10
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever],
            weights=[0.5, 0.5]
        )
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=ensemble_retriever,
            return_source_documents=True
        )
    
    def get_file_list(self):
        """Get list of uploaded files with metadata"""
        return [
            {
                'filename': filename,
                'num_documents': metadata['num_documents'],
                'num_chunks': metadata['num_chunks'],
                'file_type': metadata['file_type'],
                'file_size': metadata['file_size'],
                'file_size_formatted': metadata['file_size_formatted'],
                'upload_date': metadata.get('upload_date', 'Unknown')
            }
            for filename, metadata in self.file_metadata.items()
        ]
    
    def clear_all(self):
        """Clear all stored documents"""
        self.all_documents = []
        self.all_texts = []
        self.file_metadata = {}
        self.qdrant_vectorstore = None
        self.qa_chain = None
        self.conversation_history = []

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

class FileListResponse(BaseModel):
    files: List[dict]
    total_files: int
    total_documents: int
    total_chunks: int

@app.get("/")
def read_root():
    return {"Hello": "Multi-File RAG System with File Type Tracking"}

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

@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingests a file and adds it to the existing vector store with enhanced metadata tracking."""
    print(f"Received file for ingestion: {file.filename}")
    
    # Check if file already exists
    if file.filename in doc_store.file_metadata:
        raise HTTPException(
            status_code=400, 
            detail=f"File '{file.filename}' has already been uploaded. Please use a different name or clear existing files first."
        )
    
    temp_file_path = None
    try:
        # Create a temporary directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        temp_file_path = os.path.join("temp", file.filename)
        print(f"File saved to temporary path: {temp_file_path}")

        # Save the uploaded content to the temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get file information
        file_size = os.path.getsize(temp_file_path)
        file_type = get_file_type_from_extension(file.filename)
        
        print(f"File details - Type: {file_type}, Size: {format_file_size(file_size)}")

        # Determine the loader based on file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".pdf":
            # Using UnstructuredPDFLoader for PDFs
            loader = UnstructuredPDFLoader(temp_file_path)
        else:
            # Using UnstructuredFileLoader for other file types
            loader = UnstructuredFileLoader(temp_file_path)

        print("Starting document loading...")
        documents = loader.load()
        print("Document loading complete.")
        print(f"Loaded {len(documents)} documents.")

        # Filter out empty documents
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No readable content found in {file.filename}. The file might be empty or corrupted."
            )

        # Using RecursiveCharacterTextSplitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        print("Starting text chunking...")
        texts = text_splitter.split_documents(documents)
        print("Text chunking complete.")
        print(f"Created {len(texts)} text chunks.")

        # Filter out very short chunks
        texts = [text for text in texts if len(text.page_content.strip()) > 10]
        
        if not texts:
            raise HTTPException(
                status_code=400,
                detail=f"No meaningful text chunks created from {file.filename}."
            )

        # Prepare file info for storage
        from datetime import datetime
        file_info = {
            'type': file_type,
            'size': file_size,
            'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add documents to the store
        doc_store.add_documents(documents, texts, file.filename, file_info)

        return IngestResponse(
            message=f"File '{file.filename}' ingested successfully and added to existing knowledge base.",
            num_documents=len(documents),
            num_chunks=len(texts),
            total_files=len(doc_store.file_metadata),
            total_documents=len(doc_store.all_documents),
            total_chunks=len(doc_store.all_texts),
            file_type=file_type,
            file_size_formatted=format_file_size(file_size)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {e}")
    finally:
        # Cleanup the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat")
def chat_with_docs(request: ChatRequest):
    """Answers a user's query based on all ingested documents with enhanced source information."""
    if doc_store.qa_chain is None:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet. Please upload files first.")

    try:
        # Use the conversation history from document store
        result = doc_store.qa_chain.invoke({
            'question': request.query, 
            'chat_history': doc_store.conversation_history
        })
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

        # Update conversation history
        doc_store.conversation_history.append((request.query, response))

        return {
            "response": response,
            "source_files": source_files,
            "file_types": file_types,
            "num_sources": len(source_docs),
            "total_files_in_knowledge_base": len(doc_store.file_metadata)
        }
    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error during chat: {e}")