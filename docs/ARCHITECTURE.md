# Architecture Guide

Technical overview of Project Pulse's system design and implementation.

---

> **Document Scope**  
> This document describes the **actual implemented architecture** of this proof-of-concept system. 
> Brief production scaling considerations are included at the end but are NOT implemented.

---

## Table of Contents
- [System Overview](#system-overview)
- [Tech Stack](#tech-stack)
- [RAG Pipeline](#rag-pipeline)
- [Dual LLM Architecture](#dual-llm-architecture)
- [Data Flow](#data-flow)
- [Advanced Features](#advanced-features)
- [Production Deployment Considerations](#production-deployment-considerations)

---

## System Overview

Project Pulse is a containerized RAG system with three main components running on a single Docker host:

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Environment                   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Frontend   │  │   Backend    │  │    Ollama    │   │
│  │   (Nginx)    │  │  (FastAPI)   │  │  (Optional)  │   │
│  │              │  │              │  │              │   │
│  │  Port: 80    │  │  Port: 8000  │  │ Port: 11434  │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                  │                  │         │
│         │   API Calls      │   LLM Requests   │         │
│         └──────────────────┴──────────────────┘         │
│                            │                            │
│                    ┌───────┴────────┐                   │
│                    │  Qdrant Vector │                   │
│                    │     Storage    │                   │
│                    │  (Persistent)  │                   │
│                    └────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

**Key Characteristics:**
- Single-host deployment (Docker Compose)
- Persistent local storage (Qdrant on disk)
- In-memory BM25 index (loaded at startup)
- Optional external LLM (Groq) or local LLM (Ollama)

---

## Tech Stack

### Frontend Layer
| Component | Purpose | Version |
|-----------|---------|---------|
| **HTML/CSS/JS** | User Interface | Vanilla JavaScript (no framework) |
| **Nginx** | Web Server | Alpine-based |

### Backend Layer
| Component | Purpose | Version |
|-----------|---------|---------|
| **FastAPI** | Web Framework | 0.116+ |
| **Python** | Runtime | 3.11 |
| **LangChain** | RAG Orchestration | 0.3+ |
| **Pydantic** | Data Validation | 2.0+ |

### RAG Components
| Component | Purpose | Details |
|-----------|---------|---------|
| **Qdrant** | Vector Database | Local persistent storage, COSINE distance |
| **BM25Retriever** | Keyword Search | In-memory lexical retrieval |
| **EnsembleRetriever** | Hybrid Search | Combines BM25 (40%) + Vector (60%) |
| **CrossEncoder** | Reranking | ms-marco-MiniLM-L-6-v2 (optional) |
| **Sentence Transformers** | Embeddings | all-mpnet-base-v2 (768-dim) |

### LLM Layer
| Component | Purpose | Details |
|-----------|---------|---------|
| **Groq** | Cloud LLM | Llama 3.3 70B, fast inference |
| **Ollama** | Local LLM | Llama 3.1 8B, tested on CPU |

### Document Processing
| Component | Purpose | Details |
|-----------|---------|---------|
| **Unstructured** | Multi-format Parser | PDF, Word, Excel, PPT, HTML, etc. |
| **RecursiveCharacterTextSplitter** | Chunking | 1000 chars, 200 overlap |

---

## RAG Pipeline

### 1. Document Ingestion Flow

```
User Upload
    ↓
Validation (size, type, uniqueness)
    ↓
Temporary Storage (/temp)
    ↓
Document Loading (Unstructured)
    ↓
Text Chunking (1000 chars, 200 overlap)
    ↓
Contextual Enrichment (optional)
    ├─ Add document metadata
    ├─ Add surrounding context
    └─ Preserve original content
    ↓
Embedding Generation (768-dim vectors)
    ↓
Qdrant Upsert (UUID-based points)
    ↓
In-Memory Update (BM25 index, metadata)
    ↓
JSON Backup (documents.json, text_chunks.json)
    ↓
QA Chain Rebuild
```

**Key Implementation Details:**

**Document Loading:**
- Uses `UnstructuredPDFLoader` for PDFs
- Uses `UnstructuredFileLoader` for other formats
- Filters out empty documents

**Text Chunking:**
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```
- Recursive splitting for better semantic boundaries
- 200-char overlap to preserve context across chunks
- Filters chunks shorter than 20 characters

**Contextual Enrichment** (if enabled):
Each chunk is enriched with:
- Document title and summary
- Chunk position (e.g., "3/15")
- 200 characters of preceding context
- 200 characters of following context
- Original content preserved in metadata

**Embedding & Storage:**
- Batch embedding generation for efficiency
- Points stored with UUID identifiers
- Metadata includes: source_file, file_type, chunk_position, enrichment flag

---

### 2. Query Processing Flow

```
User Query
    ↓
Hybrid Retrieval
    ├─ BM25 (keyword matching)
    └─ Vector Search (semantic similarity)
    ↓
Combine Results (40% BM25, 60% Vector)
    ↓
[If reranking enabled]
    ↓
Cross-Encoder Scoring
    ↓
Filter by Relevance Threshold (>0.3)
    ↓
Keep Top 7 Chunks
    ↓
LLM Generation
    ├─ Build prompt with context
    ├─ Include conversation history
    └─ Generate answer
    ↓
Response with Source Attribution
```

**Key Implementation Details:**

**Hybrid Retrieval:**
```python
EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```
- Retrieves 20 candidates initially (if reranking enabled)
- BM25 catches exact keyword matches
- Vector search handles semantic similarity

**Cross-Encoder Reranking** (optional):
- Processes query + document pairs together
- Generates relevance score (0.0 to 1.0)
- Filters candidates below 0.3 threshold
- Returns top 7 most relevant chunks
- Adds `rerank_score` and `rerank_position` to metadata

**LLM Generation:**
- Uses custom prompt template emphasizing context comprehension
- Includes last 20 conversation turns for context
- Returns answer + source documents with metadata

---

## Dual LLM Architecture

### Design Philosophy

**Challenge:** Need fast development iteration AND data privacy option

**Solution:** Configurable LLM provider with automatic fallback

### Provider Comparison

| Feature | Groq (Cloud) | Ollama (Local) |
|---------|--------------|----------------|
| **Speed** | 800+ tokens/sec | Multiple minutes (CPU) |
| **Privacy** | Data sent to API | 100% local |
| **Cost** | Free tier available | Hardware cost only |
| **Setup** | API key only | Model download required |
| **Offline** | Requires internet | Works offline |
| **Testing Status** | Tested, works well | Tested, slow on CPU |

### Implementation

**Initialization Logic:**
```python
def initialize_llm():
    if LLM_PROVIDER == "ollama":
        try:
            llm = Ollama(model="llama3.1:8b", base_url="http://ollama:11434")
            llm.invoke("test", timeout=5)  # Test connection
            return llm
        except:
            logger.warning("Ollama unavailable, falling back to Groq")
    
    # Use Groq (explicit or fallback)
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
```

**Fallback Behavior:**
- Attempts configured provider first
- Automatic failover to Groq if primary fails
- All provider changes logged
- No user intervention required

**Switching Providers:**
```bash
# Edit .env
LLM_PROVIDER=ollama  # or groq

# Restart backend
docker-compose restart backend
```

---

## Data Flow

### Document Upload Data Flow

```
1. Frontend: File selection
   ↓
2. API: POST /ingest (multipart/form-data)
   ↓
3. Validation: Size, type, uniqueness checks
   ↓
4. Temporary Storage: Save to /temp
   ↓
5. Processing: Load → Chunk → Enrich → Embed
   ↓
6. Qdrant Upsert: Store vectors + metadata
   ↓
7. In-Memory Update: Add to all_documents, all_texts
   ↓
8. JSON Persistence: Backup to documents.json, text_chunks.json
   ↓
9. QA Chain Rebuild: Recreate retrieval pipeline
   ↓
10. Response: Return stats (num_chunks, processing_time, etc.)
```

### Query Data Flow

```
1. Frontend: User submits question
   ↓
2. API: POST /chat {"query": "..."}
   ↓
3. Validation: Check knowledge base not empty
   ↓
4. Hybrid Retrieval: BM25 + Vector (20 candidates)
   ↓
5. [Optional] Reranking: Cross-encoder scoring → top 7
   ↓
6. Prompt Construction: Add context + history
   ↓
7. LLM Invocation: Groq or Ollama
   ↓
8. Response Processing: Extract answer + sources
   ↓
9. History Update: Append to conversation_history
   ↓
10. Response: Answer + source_details + metadata
```

---

## Advanced Features

### 1. Contextual Chunk Enrichment

**Purpose:** Preserve document context in isolated text chunks

**Configuration:**
```env
ENABLE_CONTEXTUAL_ENRICHMENT=true
CONTEXT_WINDOW_CHARS=200
INCLUDE_DOC_SUMMARY=true
```

**What It Does:**
- Extracts document title from first meaningful line
- Generates document summary (first 500 chars)
- Adds 200 chars of preceding context
- Adds 200 chars of following context
- Preserves original content in metadata

**Example Enriched Chunk:**
```
[Document: medical_report_2024.pdf]
[Document Summary: Annual blood work results showing...]
[Chunk 3 of 15]
[Previous Context: ...vitamin levels were measured during...]

Your vitamin D level is 45 ng/mL, which is within 
the normal range of 30-100 ng/mL.

[Next Context: Vitamin B12 results show...]
```

**Benefits:**
- LLM sees chunk in broader document context
- Better understanding of isolated fragments
- Improved coherence in multi-chunk answers

**Implementation Note:**
- Original content preserved in `metadata['original_content']`
- Used by reranker (if enabled) to avoid reranking enriched text

---

### 2. Cross-Encoder Reranking

**Purpose:** Improve retrieval accuracy with two-stage search

**Configuration:**
```env
ENABLE_RERANKING=true
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_K_BEFORE_RERANK=20
RETRIEVAL_K_AFTER_RERANK=7
MIN_RELEVANCE_SCORE=0.3
```

**How It Works:**

**Stage 1: Initial Retrieval (Fast)**
- Hybrid search fetches 20 candidates
- BM25 + vector search with approximate scoring

**Stage 2: Reranking (Accurate)**
- Cross-encoder processes [query, document] pairs together
- Generates precise relevance score (0.0 to 1.0)
- Filters candidates below minimum threshold
- Returns top 7 most relevant chunks

**Why It Helps:**
- Initial retrieval is fast but approximate
- Cross-encoder is more accurate but slower
- Two-stage approach balances speed and accuracy

**Metadata Added:**
- `rerank_score`: Relevance score (float)
- `rerank_position`: Rank after reranking (int)

**Performance Impact:**
- Adds ~200-500ms to query time
- Significantly improves answer quality
- Can be disabled for faster (less accurate) responses

---

### 3. Custom Error Handling

**Design Philosophy:** Provide actionable error messages with context

**Error Code Structure:**
- `CONFIG_XXX`: Configuration errors (500)
- `FILE_XXX`: File processing errors (4xx)
- `VSTORE_XXX`: Vector store errors (5xx)
- `LLM_XXX`: LLM service errors (4xx/5xx)
- `RETRIEVAL_XXX`: Retrieval errors (4xx/5xx)
- `VALIDATION_XXX`: Request validation errors (4xx)

**Example Error Response:**
```json
{
  "error": {
    "type": "FileTooLarge",
    "message": "File size exceeds maximum allowed",
    "code": "FILE_002",
    "details": {
      "filename": "large_doc.pdf",
      "size_mb": 150,
      "max_size_mb": 100,
      "suggestion": "Please upload a file smaller than 100MB"
    },
    "timestamp": "2025-10-21T12:34:56",
    "request_id": "abc-123-def-456"
  }
}
```

**Key Features:**
- Unique error codes for debugging
- User-friendly messages
- Actionable suggestions
- Request ID tracking
- Proper HTTP status codes
- Structured logging

---

### 4. Persistent Storage

**Three-Layer Persistence:**

**Layer 1: Qdrant Vector Store**
- Location: `./qdrant_storage/`
- Contains: Vector embeddings, metadata, HNSW index
- Survives: Container restarts
- Format: Binary (Qdrant native)

**Layer 2: JSON Backups**
- `documents.json`: Original document objects
- `text_chunks.json`: All text chunks with metadata
- `file_metadata.json`: Upload history and stats
- Purpose: Rebuild capability, debugging

**Layer 3: Docker Volumes**
- `ollama_data`: Ollama models persist across restarts
- Mounted to host: Yes (except Ollama volume)

**Recovery Process:**
On application restart:
1. Qdrant client reconnects to existing storage
2. JSON backups loaded into memory
3. File metadata reconstructed
4. QA chain rebuilt from persisted data

---

### 5. Concurrent Processing

**Design:** Thread-based concurrency for CPU-intensive operations

**Thread Pool Configuration:**
```python
executor = ThreadPoolExecutor(max_workers=4)
```

**Use Cases:**
- Document processing (parsing, chunking, embedding)
- Multiple file uploads (processed in parallel)
- API requests (FastAPI handles async natively)

**Why Thread Pools:**
- CPU-bound operations (embedding generation, document parsing)
- GIL not a bottleneck (calls external libraries in C/C++)
- Simpler than multiprocessing for this use case

**Limitations:**
- Single host, limited by available CPU cores
- No distributed processing
- Suitable for demo/personal use, not high concurrency

---

## Production Deployment Considerations

**Current Architecture:** This proof-of-concept runs on a single Docker host with local persistent storage. The system demonstrates solid architecture patterns but is not production-ready.

**For Production Deployment at Scale, Would Require:**

- **Authentication & Authorization:** SSO integration (OAuth 2.0, SAML), role-based access control, document-level permissions, audit logging
- **Infrastructure Scaling:** Cloud vector store migration (Qdrant Cloud or Pinecone), distributed lexical search (Elasticsearch or OpenSearch), load-balanced LLM endpoints, Kubernetes deployment with auto-scaling
- **Security Hardening:** Secrets management (AWS Secrets Manager, HashiCorp Vault), HTTPS/TLS encryption, input sanitization, rate limiting, security compliance (GDPR, HIPAA, SOC 2)
- **Enterprise Integration:** SharePoint/Confluence/Google Workspace connectors, Slack/Teams bot interfaces, multi-tenant architecture with data isolation
- **Observability:** Centralized logging (ELK stack or similar), metrics and monitoring (Prometheus, Grafana), distributed tracing, alerting and incident response

These are architectural considerations beyond the scope of this portfolio project. The current implementation demonstrates the core RAG techniques and thoughtful code design that would form the foundation of such a system.

---

## Next Steps

- [Installation Guide](INSTALLATION.md) - Set up the system
- [API Reference](API.md) - Explore endpoints
- [Troubleshooting](TROUBLESHOOTING.md) - Fix issues

---

[← Back to Main README](../README.md)