# Architecture Notes

Technical details about how this Project Pulse RAG system works. This is more for my own documentation and anyone curious about the implementation.

## System Overview

```
┌─────────────┐
│  Frontend   │  Simple vanilla JS interface
│  (Nginx)    │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   FastAPI   │  REST API with auto docs
│   Backend   │
└──────┬──────┘
       │
       ├─────→ ┌──────────────┐
       │       │   Qdrant     │  Vector storage
       │       │ (persistent) │
       │       └──────────────┘
       │
       ├─────→ ┌──────────────┐
       │       │   BM25       │  Keyword search
       │       │  (in-memory) │
       │       └──────────────┘
       │
       └─────→ ┌──────────────┐
               │  LLM         │  Groq or Ollama
               │ (Groq/Ollama)│
               └──────────────┘
```

## RAG Pipeline

Here's what happens when you ask a question:

**1. Document Upload**
- User uploads file → saved temporarily
- Parse document (using unstructured library)
- Split into chunks (1000 chars with 200 overlap)
- Optionally enrich chunks with context
- Generate embeddings (all-mpnet-base-v2)
- Store in Qdrant vector database
- Also keep in-memory for BM25 search

**2. Query Processing**
- User asks question → convert to embedding
- Hybrid search:
  - BM25 retriever finds keyword matches
  - Vector search finds semantic matches
  - Ensemble combines both (weights: 0.4 BM25, 0.6 vector)
- Retrieve top 20 candidates

**3. Reranking (optional)**
- Use cross-encoder to score each candidate
- Filter by minimum relevance score (0.3)
- Keep top 7 after reranking

**4. Answer Generation**
- Send context + question to LLM
- LLM generates answer
- Return answer with source citations

## Key Components

### Backend (FastAPI)

**main.py** - Entry point
- Defines API endpoints
- Handles file uploads
- Manages error responses
- CORS middleware

**services.py** - Core logic
- DocumentStore class manages everything
- Document processing (loading, chunking, enrichment)
- Vector store operations
- QA chain management
- Reranking implementation

**config.py** - Configuration
- Loads environment variables
- Initializes models (LLM, embeddings, reranker)
- Sets up feature flags

**exceptions.py** - Error handling
- Custom exception classes with error codes
- Structured error responses
- Request ID tracking

**utils.py** - Helper functions
- File validation
- Size formatting
- Retry logic
- Rate limiting (basic implementation)

### Document Processing

**Chunking strategy:**
I experimented with different chunk sizes. Too small loses context, too large makes retrieval noisy. Settled on:
- Size: 1000 characters
- Overlap: 200 characters
- Min length: 20 characters (filters tiny chunks)

**Contextual enrichment:**
When enabled, each chunk gets:
- Document filename
- Chunk position (e.g., "3/15")
- 200 chars before and after
- Optional document summary

This helps the LLM understand context better.

**Supported formats:**
- PDFs, Word docs, Excel, PowerPoint
- Text files, Markdown, HTML
- CSV, JSON, XML

Uses `unstructured` library for parsing.

### Retrieval

**Hybrid search approach:**
Neither keyword nor semantic search works great alone:
- BM25 is good for exact terms but misses synonyms
- Vector search is good for concepts but misses specific terms
- Combining both gives better results

**Ensemble weights:**
Currently using 40% BM25, 60% vector. This is configurable but I found these weights work well for most queries.

**Reranking:**
The initial hybrid search gets about 20 candidates. The cross-encoder then scores each one based on relevance to the query. Typically filters out 60-70% of results, keeping only the most relevant chunks.

Without reranking, I was seeing too many false positives in the context.

### Vector Storage

**Qdrant:**
- Stores embeddings locally in `./qdrant_storage/`
- Persists across restarts
- Distance metric: cosine similarity
- Embedding dimension: 768 (from all-mpnet-base-v2)

**Why Qdrant?**
- Easy to run locally
- Good Docker support
- Fast enough for this use case
- Simple API

Considered FAISS but Qdrant's API is nicer and it handles persistence well.

### LLM Integration

**Groq:**
- Fast cloud inference
- Free tier available
- Uses llama-3.3-70b-versatile
- Temperature: 0 (deterministic)

**Ollama:**
- Runs locally
- Complete privacy
- Tested with llama3.1:8b
- Slow on CPU (3+ minutes per query)

**Prompt template:**
Basic prompt that tells the LLM to:
1. Only use provided context
2. Say if it can't answer
3. Cite sources

Nothing fancy but it works.

## Data Flow

**File Upload:**
```
File → Parse → Chunk → Enrich → Embed → Qdrant + BM25
```

**Query:**
```
Question → Embed → Hybrid Search → Rerank → LLM → Answer
```

**Metadata tracking:**
Each chunk stores:
- source_file
- file_type
- file_size
- page (if PDF)
- enriched (boolean)
- chunk_position

## Storage

**Persistent storage:**
- Qdrant data: `./backend/qdrant_storage/`
- Documents JSON: `qdrant_storage/documents.json`
- Chunks JSON: `qdrant_storage/text_chunks.json`
- Metadata: `qdrant_storage/file_metadata.json`

**Temporary storage:**
- Uploaded files: `./backend/temp/`
- Cleaned up after processing

## Performance

**What I measured:**
- File processing: ~2-5 seconds for typical PDFs
- Query latency: 1-3 seconds with Groq
- Vector search: <100ms
- Reranking: ~200-500ms for 20 candidates

**Bottlenecks:**
1. LLM inference (especially with Ollama on CPU)
2. PDF parsing for large files
3. Embedding generation on first upload

**What could be optimized:**
- Batch embedding generation
- Parallel document processing
- Cache frequent queries
- Pre-process common document types

## Scaling Considerations

This is a single-user system. If I needed to scale it:

**Database:**
- Move Qdrant to cloud instance
- Use connection pooling
- Add read replicas

**Search:**
- Add Elasticsearch for better BM25 at scale
- Implement search result caching
- Use approximate nearest neighbor search

**LLM:**
- Add request queuing
- Implement streaming responses
- Use smaller models for classification
- Cache common questions

**Architecture:**
- Add Redis for session management
- Use message queue for async processing
- Add load balancer
- Separate services (upload, query, admin)

But for now, single instance works fine.

## Security

**Current state:**
- No authentication (single user assumed)
- API keys in .env file
- CORS allows everything
- No rate limiting
- No input sanitization beyond file type checks

**For production would need:**
- User authentication (JWT tokens)
- Role-based access control
- Secrets management (not env vars)
- Rate limiting per user
- Input validation and sanitization
- HTTPS only
- Audit logging
- Data encryption at rest

This is a prototype so I skipped most of this.

## Testing

**Current testing:**
- Manual testing during development
- Basic integration tests in `tests/`
- No unit tests
- No performance tests

**Should add:**
- Unit tests for core functions
- Integration tests for API endpoints
- Performance benchmarks
- Load testing

## Tech Choices

**Why FastAPI?**
- Fast to develop with
- Auto-generates API docs
- Good async support
- Type hints everywhere

**Why LangChain?**
- Lots of integrations
- Good abstractions for RAG
- Active community
- Easy to swap components

**Why not X?**
- LlamaIndex: Considered it, but LangChain was more familiar
- ChromaDB: Qdrant seemed more mature
- Pinecone: Wanted local-first solution

## Lessons Learned

**Document chunking is hard:**
Initially used 500 char chunks - too small. Then tried 2000 - too big. 1000 with 200 overlap seems to be the sweet spot.

**Reranking is worth it:**
Added this later after seeing poor retrieval results. Made a big difference in answer quality.

**Ollama needs GPU:**
Thought I could get away with CPU. I was wrong. 3+ minutes per query is unusable.

**Error handling matters:**
Added structured error codes after dealing with cryptic FastAPI errors. Much easier to debug now.

**Logging is essential:**
Originally had minimal logging. Added structured logging with request IDs after trying to debug issues.

## Future Ideas

Things I might add:

**Better chunking:**
- Semantic chunking (split on topic boundaries)
- Overlap based on content similarity
- Preserve document structure

**Multi-modal:**
- Extract images from PDFs
- OCR for scanned documents
- Table extraction and parsing

**Better context:**
- Document metadata (author, date, etc.)
- Cross document relationships
- Citation graphs

**UI improvements:**
- Show chunk relevance scores
- Highlight source text
- Better file management

**Evaluation:**
- Automated quality metrics
- A/B testing different retrieval strategies
- User feedback collection

Maybe later. For now it works well enough.

## Questions?

If something's unclear or you want to know more about a specific component, open an issue on GitHub.