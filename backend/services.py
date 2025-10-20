"""
Business logic and services for RAG System.
Contains DocumentStore class, document processing, and QA chain management.
Now includes Cross-Encoder Reranking and Contextual Chunk Enrichment.
"""

import os
import shutil
import time
import threading
import json
import gc
import uuid
import logging
logger = logging.getLogger(__name__)

from typing import List, Dict, Any
from datetime import datetime

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    OptimizersConfig
)

from config import (
    QDRANT_STORAGE_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TEXT_SEPARATORS,
    MIN_CHUNK_LENGTH,
    RETRIEVAL_K,
    ENSEMBLE_WEIGHTS,
    MAX_CONVERSATION_HISTORY,
    EMBEDDING_VECTOR_SIZE,
    ENABLE_RERANKING,
    RERANKER_MODEL_NAME,
    RETRIEVAL_K_BEFORE_RERANK,
    RETRIEVAL_K_AFTER_RERANK,
    MIN_RELEVANCE_SCORE,
    ENABLE_CONTEXTUAL_ENRICHMENT,
    CONTEXT_WINDOW_CHARS,
    INCLUDE_DOC_SUMMARY,
    llm,
    embeddings,
    reranker
)
from exceptions import (
    QdrantConnectionFailed,
    QdrantUpsertFailed,
    CollectionNotFound,
    VectorStoreSyncError,
    FileProcessingError
)
from utils import format_file_size


# ============================================================================
# Qdrant Initialization
# ============================================================================

def initialize_qdrant_client():
    """Initialize Qdrant client with persistent storage."""
    try:
        # Ensure storage directory exists
        os.makedirs(QDRANT_STORAGE_PATH, exist_ok=True)
        
        # Create Qdrant client with persistent storage
        client = QdrantClient(path=QDRANT_STORAGE_PATH)
        
        logger.info(f"Qdrant client initialized with storage at: {QDRANT_STORAGE_PATH}")
        return client
    except Exception as e:
        logger.error(f"Error initializing Qdrant client: {e}")
        raise


def ensure_collection_exists(
    client: QdrantClient, 
    collection_name: str, 
    vector_size: int = EMBEDDING_VECTOR_SIZE
):
    """Ensure the collection exists, create if it doesn't."""
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            logger.info(f"Creating new collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                # FIX: Explicitly pass default OptimizersConfig to satisfy Pydantic v2 strictness
                optimizers_config=OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=100,
                    default_segment_number=0,
                    flush_interval_sec=5
                )
            )
            logger.info(f"Collection '{collection_name}' created successfully")
        else:
            logger.info(f"Collection '{collection_name}' already exists")
            
    except Exception as e:
        logger.error(f"Error managing collection: {e}")
        raise


# ============================================================================
# Document Serialization (for JSON persistence)
# ============================================================================

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
        logger.error(f"Error saving documents to {file_path}: {e}")
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
        logger.warning(f"Error loading documents from {file_path}: {e}")
        return []


def save_metadata_to_file(file_metadata: dict):
    """Save file metadata to persistent storage."""
    try:
        metadata_path = os.path.join(QDRANT_STORAGE_PATH, "file_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(file_metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")


def load_metadata_from_file() -> dict:
    """Load file metadata from persistent storage."""
    try:
        metadata_path = os.path.join(QDRANT_STORAGE_PATH, "file_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading metadata: {e}")
    return {}


# ============================================================================
# Contextual Chunk Enrichment Functions
# ============================================================================

def extract_document_metadata(filename: str, documents: List[Document]) -> dict:
    """
    Extract metadata from the full document for enrichment.
    
    Args:
        filename: Name of the source file
        documents: List of loaded documents
        
    Returns:
        Dictionary with document-level metadata
    """
    # Generate document summary (first 500 chars)
    full_text = " ".join([doc.page_content for doc in documents])
    doc_summary = full_text[:500].strip() + "..." if len(full_text) > 500 else full_text
    
    # Extract potential title (first meaningful line)
    lines = full_text.split('\n')
    potential_title = None
    for line in lines[:10]:  # Check first 10 lines
        if line.strip() and len(line.strip()) > 10:
            potential_title = line.strip()[:100]
            break
    
    return {
        'filename': filename,
        'title': potential_title or filename,
        'summary': doc_summary,
        'total_length': len(full_text),
        'num_source_docs': len(documents)
    }


def enrich_chunk_with_context(
    chunk: Document,
    all_chunks: List[Document],
    chunk_index: int,
    doc_metadata: dict
) -> Document:
    """
    Enrich a text chunk with surrounding context and document metadata.
    
    Args:
        chunk: The chunk to enrich
        all_chunks: All chunks from the document
        chunk_index: Index of current chunk
        doc_metadata: Document-level metadata
        
    Returns:
        Enriched Document with added context
    """
    enriched_content = []
    
    # Add document context header
    enriched_content.append(f"[Document: {doc_metadata['filename']}]")
    
    if INCLUDE_DOC_SUMMARY and chunk_index == 0:
        enriched_content.append(f"[Document Summary: {doc_metadata['summary']}]")
    
    enriched_content.append(f"[Chunk {chunk_index + 1} of {len(all_chunks)}]")
    
    # Add preceding context (if available)
    if chunk_index > 0:
        prev_chunk = all_chunks[chunk_index - 1]
        prev_context = prev_chunk.page_content[-CONTEXT_WINDOW_CHARS:].strip()
        if prev_context:
            enriched_content.append(f"[Previous Context: ...{prev_context}]")
    
    # Add the main chunk content
    enriched_content.append("")
    enriched_content.append(chunk.page_content)
    enriched_content.append("")
    
    # Add following context (if available)
    if chunk_index < len(all_chunks) - 1:
        next_chunk = all_chunks[chunk_index + 1]
        next_context = next_chunk.page_content[:CONTEXT_WINDOW_CHARS].strip()
        if next_context:
            enriched_content.append(f"[Next Context: {next_context}...]")
    
    # Create enriched document
    enriched_doc = Document(
        page_content="\n".join(enriched_content),
        metadata={
            **chunk.metadata,
            'original_content': chunk.page_content,  # Keep original for reference
            'enriched': True,
            'doc_title': doc_metadata['title'],
            'chunk_position': f"{chunk_index + 1}/{len(all_chunks)}"
        }
    )
    
    return enriched_doc


def apply_contextual_enrichment(
    documents: List[Document],
    texts: List[Document],
    filename: str
) -> List[Document]:
    """
    Apply contextual enrichment to all text chunks.
    
    Args:
        documents: Original loaded documents
        texts: Text chunks to enrich
        filename: Source filename
        
    Returns:
        List of enriched text chunks
    """
    if not ENABLE_CONTEXTUAL_ENRICHMENT:
        return texts
    
    logger.debug(f"[ENRICHMENT] Applying contextual enrichment to {len(texts)} chunks from {filename}")
    
    # Extract document-level metadata
    doc_metadata = extract_document_metadata(filename, documents)
    
    # Enrich each chunk
    enriched_texts = []
    for i, chunk in enumerate(texts):
        enriched_chunk = enrich_chunk_with_context(chunk, texts, i, doc_metadata)
        enriched_texts.append(enriched_chunk)
    
    logger.debug(f"[ENRICHMENT] Enrichment complete. Sample enriched chunk length: {len(enriched_texts[0].page_content) if enriched_texts else 0} chars")
    
    return enriched_texts


# ============================================================================
# Reranking Functions
# ============================================================================

def rerank_documents(query: str, documents: List[Document]) -> List[Document]:
    """
    Rerank retrieved documents using cross-encoder model.
    
    Args:
        query: The search query
        documents: Retrieved documents to rerank
        
    Returns:
        Reranked and filtered documents
    """
    if not ENABLE_RERANKING or reranker is None or not documents:
        return documents[:RETRIEVAL_K_AFTER_RERANK]
    
    logger.debug(f"[RERANKING] Reranking {len(documents)} documents for query: '{query[:50]}...'")
    
    try:
        # Prepare query-document pairs for reranking
        # Use original content if available (before enrichment)
        pairs = [
            [query, doc.metadata.get('original_content', doc.page_content)] 
            for doc in documents
        ]
        
        # Get relevance scores
        scores = reranker.predict(pairs)
        
        # Combine documents with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum relevance score
        filtered_docs = [
            (doc, score) for doc, score in doc_scores 
            if score >= MIN_RELEVANCE_SCORE
        ]
        
        logger.debug(f"[RERANKING] Filtered to {len(filtered_docs)} documents above threshold {MIN_RELEVANCE_SCORE}")
        
        if filtered_docs:
            logger.debug(f"[RERANKING] Score range: {filtered_docs[0][1]:.3f} (best) to {filtered_docs[-1][1]:.3f} (worst)")
        
        # Take top K after reranking
        reranked_docs = [doc for doc, score in filtered_docs[:RETRIEVAL_K_AFTER_RERANK]]
        
        # Add reranking score to metadata for debugging
        for i, (doc, score) in enumerate(filtered_docs[:RETRIEVAL_K_AFTER_RERANK]):
            reranked_docs[i].metadata['rerank_score'] = float(score)
            reranked_docs[i].metadata['rerank_position'] = i + 1
        
        return reranked_docs
        
    except Exception as e:
        logger.error(f"[RERANKING] Error during reranking: {e}")
        return documents[:RETRIEVAL_K_AFTER_RERANK]


# ============================================================================
# Custom Reranking Retriever
# ============================================================================

# Replace the RerankingRetriever class in services.py (around line 300)

class RerankingRetriever(BaseRetriever):
    """
    Custom retriever that applies cross-encoder reranking to results.
    """
    
    base_retriever: BaseRetriever
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Get documents relevant to query, with reranking."""
        # Get initial documents from base retriever
        # Don't pass run_manager if it's causing issues
        docs = self.base_retriever.get_relevant_documents(query)
        
        # Apply reranking
        reranked_docs = rerank_documents(query, docs)
        
        return reranked_docs
    
    async def _aget_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async version - falls back to sync for now."""
        return self._get_relevant_documents(query, run_manager=run_manager)


# ============================================================================
# Document Processing
# ============================================================================

def process_document_sync(temp_file_path: str, filename: str) -> tuple:
    """
    Synchronous document processing function to run in thread pool.
    Now includes contextual enrichment.
    
    Args:
        temp_file_path: Path to temporary file
        filename: Original filename
        
    Returns:
        Tuple of (documents, texts, processing_time)
        
    Raises:
        FileProcessingError: If document processing fails
    """
    start_time = time.time()
    
    try:
        # Determine the loader based on file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            loader = UnstructuredPDFLoader(temp_file_path)
        else:
            loader = UnstructuredFileLoader(temp_file_path)

        logger.info(f"Starting document loading for {filename}...")
        documents = loader.load()
        logger.info(f"Document loading complete for {filename}. Loaded {len(documents)} documents.")

        # Filter out empty documents
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        if not documents:
            raise ValueError(f"No readable content found in {filename}. The file might be empty or corrupted.")

        # Using RecursiveCharacterTextSplitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=TEXT_SEPARATORS
        )

        logger.info(f"Starting text chunking for {filename}...")
        texts = text_splitter.split_documents(documents)
        logger.info(f"Text chunking complete for {filename}. Created {len(texts)} text chunks.")

        # Filter out very short chunks
        texts = [text for text in texts if len(text.page_content.strip()) > MIN_CHUNK_LENGTH]
        
        if not texts:
            raise ValueError(f"No meaningful text chunks created from {filename}.")
        
        # Apply contextual enrichment
        texts = apply_contextual_enrichment(documents, texts, filename)

        processing_time = time.time() - start_time
        return documents, texts, processing_time
        
    except Exception as e:
        import traceback
        logger.error(f"File processing failed for {filename}: {e}")
        logger.error(f"DETAILED ERROR for {filename}:")
        logger.error(f"{traceback.format_exc()}")
        raise FileProcessingError(filename, str(e))


# ============================================================================
# Document Store Class
# ============================================================================

class DocumentStore:
    """
    Manages document storage, vector store, and QA chain.
    Thread-safe with persistent storage support.
    Now includes reranking and contextual enrichment.
    """
    
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
                logger.debug(f"Found {len(self.file_metadata)} files in persistent storage")
                logger.debug(f"Loaded {len(self.all_documents)} documents and {len(self.all_texts)} text chunks")
                self._rebuild_qa_chain()
            else:
                logger.info("No existing data found, starting fresh")
                
        except Exception as e:
            logger.warning(f"Error initializing persistent storage: {e}")
            logger.warning("Falling back to in-memory storage")
            self.qdrant_client = None

    def ensure_client_is_ready(self):
        """Checks if the Qdrant client is None and re-initializes it if needed."""
        if self.qdrant_client is None:
            logger.info("[RE-INIT] Client is None. Attempting to re-initialize Qdrant client and collection...")
            try:
                self.qdrant_client = initialize_qdrant_client()
                ensure_collection_exists(self.qdrant_client, COLLECTION_NAME)
                logger.info("[RE-INIT] Qdrant client and collection successfully re-initialized.")
            except Exception as e:
                logger.error(f"[RE-INIT] Error re-initializing Qdrant client: {e}")
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
            
            logger.debug(f"Saved complete state: {len(self.all_documents)} documents, {len(self.all_texts)} text chunks")
            
        except Exception as e:
            logger.error(f"Error saving complete state: {e}")
    
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
            logger.warning(f"Error loading complete state: {e}")
            self.all_documents = []
            self.all_texts = []
            self.file_metadata = {}
    
    def add_documents(self, documents, texts, filename, file_info):
        """Add new documents and texts to the store with thread safety and persistence"""
        with self._lock:
            # Ensure client is ready before attempting ingestion
            self.ensure_client_is_ready() 
            
            if self.qdrant_client is None:
                raise QdrantConnectionFailed(QDRANT_STORAGE_PATH, "Client initialization failed")

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
                logger.debug(f"Starting Qdrant upsert process for {len(texts)} chunks")
                
                try:
                    # Generate vectors
                    text_contents = [text.page_content for text in texts]
                    vectors = embeddings.embed_documents(text_contents)
                    
                    if not vectors or len(vectors) != len(texts):
                        raise QdrantUpsertFailed(COLLECTION_NAME, len(texts), "Vector generation failed or vector count mismatch")

                    logger.debug(f"Vectorization complete. Created {len(vectors)} vectors")

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
                    logger.debug(f"Calling qdrant_client.upsert for {len(points)} points")
                    self.qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                        wait=True
                    )
                    logger.info(f"Successfully upserted {len(points)} points to Qdrant.")
                    
                    # Verify count
                    final_count_result = self.qdrant_client.count(COLLECTION_NAME, exact=True)
                    logger.debug("--- QDRANT PERSISTENCE CHECK ---") # Changed
                    logger.debug(f"Upsert Complete. Current Qdrant Count: {final_count_result.count}") # Changed
                    logger.debug("--------------------------------")
                    
                except QdrantUpsertFailed:
                    # Re-raise our custom exception
                    raise
                except Exception as e:
                    logger.critical(f"!!! CRITICAL QDRANT ERROR during upsert: {e}")
                    raise QdrantUpsertFailed(COLLECTION_NAME, len(points) if 'points' in locals() else len(texts), str(e))
            
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

            if self.qdrant_client is None:
                raise QdrantConnectionFailed(QDRANT_STORAGE_PATH, "Client not available")

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
                logger.error(f"Error removing file {filename}: {e}")
                raise VectorStoreSyncError("delete", str(e))
    
    def _create_qa_prompt(self):
        """Create an enhanced prompt for better multi-document retrieval"""
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
        """Rebuild the QA chain with all documents and reranking support"""
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
            
            # Adjust retrieval K based on whether reranking is enabled
            retrieval_k = RETRIEVAL_K_BEFORE_RERANK if ENABLE_RERANKING else RETRIEVAL_K
            
            if self.all_texts and self.qdrant_vectorstore:
                qdrant_retriever = self.qdrant_vectorstore.as_retriever(
                    search_kwargs={'k': retrieval_k}
                )
                
                bm25_retriever = BM25Retriever.from_documents(documents=self.all_texts)
                bm25_retriever.k = retrieval_k
                
                base_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, qdrant_retriever],
                    weights=ENSEMBLE_WEIGHTS
                )
            
            elif self.qdrant_vectorstore:
                base_retriever = self.qdrant_vectorstore.as_retriever(
                    search_kwargs={'k': retrieval_k}
                )
            else:
                self.qa_chain = None
                return
            
            # Wrap retriever with reranking if enabled
            if ENABLE_RERANKING:
                logger.info("QA chain will use cross-encoder reranking")
                # Create a custom retriever that applies reranking
                final_retriever = RerankingRetriever(base_retriever=base_retriever)
            else:
                final_retriever = base_retriever
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=final_retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={
                    "prompt": self._create_qa_prompt()
                }
            )
            
        except Exception as e:
            logger.error(f"Error rebuilding QA chain: {e}")
            self.qa_chain = None
    
    def has_documents(self) -> bool:
        """Check if the store has any documents."""
        with self._lock:
            return len(self.all_texts) > 0
    
    def get_file_list(self) -> List[Dict[str, Any]]:
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
                    logger.critical(f"Resetting persistent Qdrant/JSON storage at {QDRANT_STORAGE_PATH}...")
                    
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
                                logger.info("Persistent storage successfully deleted.")
                            except PermissionError as pe:
                                # If still locked, try deleting individual files
                                logger.warning(f"Permission error deleting folder, attempting file-by-file deletion: {pe}")
                                for root, dirs, files in os.walk(QDRANT_STORAGE_PATH, topdown=False):
                                    for name in files:
                                        try:
                                            os.remove(os.path.join(root, name))
                                        except Exception as e:
                                            logger.warning(f"Could not delete file {name}: {e}")
                                    for name in dirs:
                                        try:
                                            os.rmdir(os.path.join(root, name))
                                        except Exception as e:
                                            logger.warning(f"Could not delete directory {name}: {e}")
                                # Try one more time to remove the root
                                try:
                                    shutil.rmtree(QDRANT_STORAGE_PATH)
                                except:
                                    logger.warning("Some files may remain locked. They will be overwritten on next initialization.")
                        
                        # --- STEP 4: Immediately re-initialize with fresh storage ---
                        logger.info("Re-initializing fresh Qdrant client and collection...")
                        self._initialize_persistent_storage()
                        logger.info("Fresh Qdrant storage initialized successfully.")
                        
                    except Exception as e:
                        logger.error(f"Error during storage reset: {e}")
                        # Ensure client is None so it can be re-initialized
                        self.qdrant_client = None
                        self.qdrant_vectorstore = None
                        # Try to initialize fresh storage anyway
                        try:
                            self._initialize_persistent_storage()
                        except Exception as init_error:
                            logger.error(f"Failed to re-initialize storage: {init_error}")
                
            except Exception as e:
                logger.error(f"Error clearing all data: {e}")

    def get_stats(self) -> Dict[str, Any]:
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