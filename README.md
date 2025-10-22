# Project Pulse / Virtual Buddy

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A privacy first RAG system for querying personal documents locally.**

> Built on Windows 11 + Docker Desktop. This is a personal project, not production hardened.

![Demo](docs/images/ProjectPulseDemo.gif)

## What is Project Pulse?

A working proof-of-concept RAG (Retrieval-Augmented Generation) system. Upload your documents (PDFs, Word files, spreadsheets), ask questions in natural language, and get accurate answers all while keeping your data private and local.

## Background

Built this to solve a personal problem - I had years of personal records scattered across PDFs and couldn't quickly find information like "what was my vitamin D level last year?"

Started simple, then kept adding features as I learned more about RAG:
- Initially just basic vector search
- Added BM25 after reading about hybrid retrieval
- Discovered reranking improves accuracy significantly
- Ollama was painfully slow on my i7 laptop (no GPU) - switched to Groq

This is a learning project, so there's definitely room for improvement!

## What I Built

This started as a weekend project to understand how RAG systems work. I wanted to build something that could actually answer questions about my personal documents without sending them to some cloud service.

**Main features:**
- Upload PDFs, Word docs, spreadsheets, and other files
- Ask questions in natural language
- Get answers with source citations
- Everything runs locally (optional - can use Groq API for faster responses)
- Persistent storage so your data survives restarts

**Tech I used:**
- FastAPI for the backend
- LangChain for the RAG pipeline
- Qdrant for vector storage
- Groq or Ollama for the LLM
- Vanilla JS for the frontend (no frameworks - keeping it simple)

## Quick Start

**Prerequisites:**
- Docker Desktop
- A Groq API key (free signup at https://console.groq.com)

**Get it running:**

```bash
git clone https://github.com/vmvadivel/Project-Pulse.git
cd Project-Pulse

# create .env file
echo 'GROQ_API_KEY="gsk_your_key_here"' > .env
echo 'LLM_PROVIDER=groq' >> .env

# start everything
docker-compose up -d
```

That's it. Go to http://localhost and start uploading files.

**Detailed setup:** Check [docs/SETUP.md](docs/SETUP.md) if you run into issues.

## Why I Built This

I was curious about how RAG systems work. Wanted to understand:
- How do you chunk documents effectively?
- How does vector search actually work?
- What's the difference between semantic and keyword search?
- Can I make this work on my laptop without spending money?

Ended up learning a lot more than I expected. The hybrid search (BM25 + vector) was particularly interesting - neither approach works great alone but combining them gives much better results.

## Features I Added

**Core stuff:**
- Multi-format document support (PDF, Word, Excel, PowerPoint, Markdown, etc.)
- Hybrid retrieval combining keyword and semantic search
- Persistent storage with Qdrant
- Works with both Groq (cloud, fast) and Ollama (local, private)

**RAG improvements:**
- Cross-encoder reranking to filter out irrelevant results
- Contextual chunk enrichment (adds surrounding text to each chunk)
- Conversation history for follow-up questions

**Code quality:**
- Custom error handling with error codes
- Structured logging
- Docker setup for easy deployment
- FastAPI with auto-generated docs

## How It Works

```
Upload docs → Parse & chunk → Generate embeddings → Store in Qdrant
                                                           ↓
                            ← Generate answer ← Rerank ← Hybrid search
                                   ↓
                            Return with sources
```

The interesting part is the hybrid search. I'm using:
1. BM25 for keyword matching (finds exact terms)
2. Vector search for semantic similarity (finds concepts)
3. Ensemble retriever to combine both with configurable weights
4. Cross-encoder reranking to score the final candidates

More details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) if you're interested.

## Project Status

**What works:**
- ✅ Basic RAG pipeline
- ✅ Multiple file formats
- ✅ Persistent storage
- ✅ Docker deployment
- ✅ Hybrid retrieval + reranking

**What needs work:**
- No authentication (single user only)
- Tested only on Windows 11 with Docker
- No automated tests
- Frontend is pretty basic
- Performance not optimized

**Known issues:**
- Ollama on CPU is extremely slow (tested it, not practical without GPU)
- Large PDFs can take a while to process
- CORS is wide open (fine for local use, bad for production)

This is a **portfolio/learning project**, not production software.

## Running Without Docker

If you want to run it locally:

```bash
# setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r backend/requirements-frozen.txt

# configure
set GROQ_API_KEY=your_key
set LLM_PROVIDER=groq

# run
cd backend
uvicorn main:app --reload
```

Then open `frontend/index.html` in your browser.

## What I Learned

**Vector databases are interesting:**
I initially thought they were just fancy databases, but understanding how embeddings work and how similarity search scales changed my perspective. The trade-offs between accuracy and speed are non-trivial.

**RAG is harder than it looks:**
Chunking strategy matters a lot. Too small and you lose context, too big and retrieval gets noisy. I experimented with different sizes and overlap ratios - ended up with 1000 chars with 200 char overlap.

**Reranking makes a big difference:**
The cross-encoder reranker filters out about 60% of initially retrieved chunks. Without it, the LLM gets too much irrelevant context and the answers suffer.

**Ollama is great for privacy but needs GPU:**
Tested Ollama on my CPU - took 3+ minutes per query. With GPU it should be much faster but I don't have one to test with.

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Backend | FastAPI | Fast, auto-docs, modern Python |
| RAG | LangChain | Lots of integrations, good abstractions |
| Vector DB | Qdrant | Works locally, easy Docker setup |
| LLM | Groq/Ollama | Groq for speed, Ollama for privacy |
| Embeddings | all-mpnet-base-v2 | Good balance of quality and speed |
| Reranker | ms-marco-MiniLM | Fast, good enough for this use case |
| Frontend | Vanilla JS | Keeping it simple, no build step |

## Use Cases

**What I use it for:**
- Searching through my medical records
- Finding specific clauses in contracts
- Querying multiple research papers at once

**What it could be used for:**
- Personal knowledge base
- Document Q&A for small teams
- Research paper analysis
- Meeting notes search

## Privacy

**Current setup:**
- All documents stored locally in Qdrant
- Can use Ollama for completely offline operation
- No telemetry or tracking

**What's not secure:**
- No authentication
- API keys in .env file
- CORS allows everything
- No rate limiting

Don't deploy this to the internet as-is. It's designed for local use.

## Things That Could Be Added

Some ideas I had but didn't implement:
- [ ] User authentication
- [ ] Multiple knowledge bases
- [ ] Better frontend UI
- [ ] Automated tests
- [ ] Performance monitoring
- [ ] Citation quality scoring
- [ ] Document preprocessing (OCR, table extraction)

Maybe I'll add these later if I need them.

## License

MIT - do whatever you want with it.

## Contact

Built by Vadivel Mohanakrishnan  
LinkedIn: https://www.linkedin.com/in/vmvadivel/

If you build something cool with this, let me know!

---

**Note:** This is a personal project. Code quality is decent but it's not production ready. Use it to learn, experiment, or build something better.