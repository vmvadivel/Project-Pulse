# Project Pulse / Virtual Buddy

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A privacy-first RAG system for querying personal documents locally.**

> **⚠️ Project Status**  
> **Type:** Personal portfolio project (proof-of-concept)  
> **Tested On:** Windows 11 + Docker Desktop  
> **Production Ready:** No - demonstrates architecture and techniques, not production hardened

---

## What Is This?

A working proof-of-concept RAG (Retrieval-Augmented Generation) system built with production grade coding practices. Upload your documents (PDFs, Word files, spreadsheets), ask questions in natural language, and get accurate answers all while keeping your data private and local.

**Built to demonstrate:**
- Advanced retrieval techniques (hybrid search, reranking, contextual enrichment)
- Clean code architecture with proper error handling and logging
- Thoughtful design patterns that consider enterprise scaling needs

**What this is NOT:**
- A production ready enterprise application
- Tested on multiple platforms (Mac/Linux commands untested)
- A commercial product

---

## Demo

![Demo](docs/images/ProjectPulseDemo.gif)

---

## Quick Start

### Prerequisites
- **Docker Desktop** ([Install](https://docs.docker.com/get-docker/))
- **Windows 11** (or Windows 10 with WSL2)
- **8GB+ RAM** (16GB recommended for Ollama)
- **Groq API Key** ([Free signup](https://console.groq.com))

### Installation (3 Steps)

**1. Clone & Configure**
```bash
git clone https://github.com/vmvadivel/Project-Pulse.git
cd Project-Pulse

# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
echo "LLM_PROVIDER=groq" >> .env
```

**2. Start with Docker**
```bash
docker-compose up -d
```

**3. Access the Application**
- **Frontend**: http://localhost
- **API Docs**: http://localhost:8000/docs

> **Detailed setup:** See [Installation Guide](docs/INSTALLATION.md)

---

## Features

### Implemented & Working

**Core Functionality**
- Multi-format document support (PDF, Word, Excel, PowerPoint, Markdown, HTML, CSV, JSON, XML)
- Natural language querying across all uploaded documents
- Persistent vector storage (Qdrant) - survives restarts
- Dual LLM support: Groq (cloud, fast) or Ollama (local, private)
- Individual file management (upload, delete, list)

**Code Quality**
- Custom error handling with unique error codes and request tracking
- Structured logging with configurable debug mode
- Thread-based concurrent processing
- RESTful API with automatic OpenAPI documentation
- Full Docker containerization

### Advanced RAG Techniques

**Hybrid Retrieval**
- Combines BM25 (keyword matching) and vector search (semantic similarity)
- Ensemble retriever with configurable weights

**Cross-Encoder Reranking** (Optional, toggleable)
- Two-stage retrieval: initial search → relevance scoring
- Filters candidates by minimum relevance threshold
- Uses ms-marco-MiniLM-L-6-v2 model

**Contextual Chunk Enrichment** (Optional, toggleable)
- Adds document metadata to each text chunk
- Includes surrounding context (200 chars before/after)
- Preserves chunk position and document summary

### Scaling Considerations (Not Implemented)

These are architectural considerations for future enterprise deployment:
- Authentication & authorization (SSO, RBAC)
- Cloud vector store migration (Qdrant Cloud/Pinecone)
- Distributed lexical search (Elasticsearch/OpenSearch)
- Enterprise connectors (SharePoint, Confluence, Google Workspace)
- Multi-tenant architecture with data isolation

*See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed scaling discussion.*

---

## Demo

![Demo GIF](docs/images/demo.gif)

**What the demo shows:**
1. Uploading multiple documents
2. Asking questions across documents
3. Receiving accurate, source-attributed answers
4. Managing files (view, delete)

---

## How It Works

```
┌─────────────┐
│   Upload    │  1. User uploads documents
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Document   │  2. Parse, chunk, and enrich
│ Processing  │     with contextual metadata
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Qdrant    │  3. Generate embeddings and
│   Vector    │     store in vector database
│   Store     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Hybrid    │  4. User query triggers
│  Retrieval  │     BM25 + vector search
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Cross-    │  5. Rerank candidates with
│  Encoder    │     cross-encoder scoring
│  Reranking  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     LLM     │  6. Generate answer using
│  (Groq/     │     top-ranked context
│   Ollama)   │
└─────────────┘
```

> **Architecture details:** See [Architecture Guide](docs/ARCHITECTURE.md)

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | HTML, CSS, JavaScript (Vanilla), Nginx |
| **Backend** | FastAPI, Python 3.11 |
| **RAG Pipeline** | LangChain, BM25Retriever, Cross-Encoder |
| **Vector DB** | Qdrant (local/persistent) |
| **LLM** | Groq (cloud), Ollama (local) |
| **Embeddings** | Sentence Transformers (all-mpnet-base-v2) |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| **Deployment** | Docker, Docker Compose |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALLATION.md) | Detailed setup and configuration (Windows + Docker tested) |
| [Architecture Guide](docs/ARCHITECTURE.md) | System design, RAG pipeline, scaling considerations |
| [API Reference](docs/API.md) | Quick reference (see `/docs` for interactive Swagger UI) |
| [Docker Reference](docs/DOCKER.md) | Container management and commands |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

**Note:** Mac/Linux commands in documentation are provided for reference but have not been personally tested. Community contributions welcome!

---

## Use Cases

### Personal
- **Medical Records**: Query health history across multiple test reports
- **Legal Documents**: Search contracts and legal correspondence
- **Research Papers**: Extract insights from academic publications
- **Personal Finance**: Analyze financial documents and invoices

### Potential Enterprise Applications
The architecture patterns demonstrated here could be adapted for:
- Internal knowledge management systems
- Customer support documentation retrieval
- Compliance and policy document search
- Employee onboarding assistance

*Note: Enterprise deployment would require additional work on authentication, security, scalability, and integration.*

---

## Privacy & Security

**Current Implementation:**
- ✅ Local vector storage (Qdrant on disk)
- ✅ Optional local LLM (Ollama)
- ✅ Data stays on your machine (with Ollama)
- ⚠️ No authentication (single-user mode)
- ⚠️ API keys in `.env` file
- ⚠️ CORS allows all origins

**For Production Use, Would Need:**
- Authentication & authorization
- Secrets management (not environment variables)
- HTTPS/TLS encryption
- Rate limiting
- Input sanitization
- Audit logging

> **Security details:** See [Architecture Guide - Security](docs/ARCHITECTURE.md#security-considerations)

---

## Development

Want to run locally without Docker?

```bash
# Setup virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
cd backend
pip install -r requirements-frozen.txt

# Run backend
set GROQ_API_KEY=your_key
uvicorn main:app --reload
```

> **Development guide:** See [Development Guide](docs/DEVELOPMENT.md)

---

## Roadmap

**Completed:**
- [x] Basic RAG implementation
- [x] Hybrid retrieval (BM25 + vector)
- [x] Cross-encoder reranking
- [x] Contextual chunk enrichment
- [x] Persistent storage
- [x] Custom error handling
- [x] Docker deployment

**Future Considerations:**
- [ ] Authentication & authorization
- [ ] Multi-platform testing (Mac, Linux)
- [ ] Cloud vector store integration
- [ ] Enterprise connectors
- [ ] Automated testing suite
- [ ] Performance benchmarking

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

This is a personal portfolio project, but suggestions and feedback are welcome! 

If you find issues or have ideas:
- Open an issue describing the problem/suggestion
- For Mac/Linux users: PRs with tested platform-specific fixes appreciated

---

## Contact

**Developer**: Vadivel Mohanakrishnan  
**LinkedIn**: https://www.linkedin.com/in/vmvadivel/    

---

## Acknowledgments

Built with:
- [LangChain](https://www.langchain.com/) for RAG orchestration
- [Groq](https://groq.com/) for fast cloud LLM inference
- [Ollama](https://ollama.ai/) for local LLM support
- [Qdrant](https://qdrant.tech/) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework

---

## Learning Resources

If you are interested in building similar systems, check out:
- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Qdrant Vector Database Guide](https://qdrant.tech/documentation/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

---

<div align="center">

**If you find this project useful or interesting, please consider giving it a ⭐!**

[📖 Documentation](docs/) • [🐛 Report Issue](issues) • [💡 Suggest Feature](issues)

</div>