# Enhanced Multi-File RAG System with Dual LLM Support

A high performance Retrieval Augmented Generation (RAG) system that enables question answering across multiple documents using advanced hybrid retrieval and persistent vector storage. It has flexible LLM deployment with both cloud based (Groq) and local (Ollama) options.

## What is This Project?

This RAG system allows you to:
- **Upload multiple documents** in various formats (PDF, Word, Excel, PowerPoint, HTML, Markdown, and more)
- **Ask questions** and get answers by retrieving relevant information across all uploaded documents
- **Hybrid search** combining semantic similarity (vector search) and keyword matching (BM25) for better retrieval quality
- **Persistent storage** ensuring your knowledge base survives restarts
- **Choose your LLM provider** use fast cloud based Groq or privacy focused local Ollama
- **Automatic fallback** between LLM providers for maximum reliability

### Key Features

- **Dual LLM Support**: Seamlessly switch between Groq (cloud) and Ollama (local) LLMs
- **Persistent Vector Storage**: Qdrant based vector database with disk persistence
- **Hybrid Retrieval**: Combines BM25 (keyword) and semantic search using ensemble retriever
- **Multi-Format Support**: PDF, Word, Excel, PowerPoint, HTML, Markdown, JSON, XML, CSV, TXT
- **Concurrent Processing**: Thread pool executor for efficient document ingestion
- **Individual File Management**: Upload, delete specific files, or clear all documents
- **Real time Statistics**: Monitor system performance, storage, and processing metrics
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation
- **Docker Deployment**: Fully containerized application with docker compose orchestration

## Tech Stack

### Backend
- **FastAPI**: Modern, high performance Python web framework
- **LangChain**: RAG orchestration and LLM integration
- **Qdrant**: Vector database for semantic search with persistent storage
- **Groq**: Cloud based LLM inference (primary for CPU systems)
- **Ollama**: Local LLM runtime for privacy and offline operation
- **HuggingFace Transformers**: Sentence embeddings (all-mpnet-base-v2)
- **Unstructured**: Multi-format document parsing and processing

### Frontend
- **HTML/CSS/JavaScript**: Vanilla JS single-page application
- **Nginx**: Web server for static file serving

### Infrastructure
- **Docker & Docker Compose**: Containerized deployment
- **Python 3.11**: Core runtime environment

## Getting Started

### Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux) - [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** v2.0 or higher
- **8GB+ RAM** available for Docker (16GB recommended if using Ollama)
- **10GB+ free disk space** (for Ollama models if used)
- **Groq API Key** - Free at [console.groq.com](https://console.groq.com) (required for Groq LLM)

### Installation & Setup

#### 1. Clone the Repository

#### 2. Create a `.env` file in the root directory and Configure the Environment Variables

**Option A: Using Groq (Recommended for CPU only systems)**
```env
GROQ_API_KEY="your_groq_api_key_here"
LLM_PROVIDER=groq
```

**Option B: Using Ollama (For local/private deployment)**
```env
GROQ_API_KEY="your_groq_api_key_here"  # Still required as fallback
LLM_PROVIDER=ollama

# Ollama configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.1:8b
```

> **Note**: The `GROQ_API_KEY` is always required as a fallback even when using Ollama, ensuring the system remains operational if Ollama becomes unavailable.

#### 3. Start the Application with Docker

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

Expected output:
```
NAME            STATUS                   PORTS
rag_backend     Up (healthy)            0.0.0.0:8000->8000/tcp
rag_frontend    Up                      0.0.0.0:80->80/tcp
rag_ollama      Up                      0.0.0.0:11434->11434/tcp
```

#### 4. Pull Ollama Model (If Using Ollama)

```bash
# Download the Ollama model (takes 3-5 minutes, ~4.7GB)
docker exec rag_ollama ollama pull llama3.1:8b

# Verify model is available
docker exec rag_ollama ollama list

# Restart backend to establish connection
docker-compose restart backend
```

#### 5. Access the Application

- **Frontend Application**: http://localhost
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Ollama API** (if enabled): http://localhost:11434

### Basic Usage

1. **Upload Documents**: Click "Upload Document" and select your file(s)
2. **View Files**: See all uploaded documents with metadata in the "Uploaded Files" section
3. **Ask Questions**: Type your question in the chat interface
4. **Get Answers**: The system retrieves relevant context and generates accurate responses
5. **Manage Files**: Delete individual files or clear all documents as needed

## Docker Commands Reference

### Service Management

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Stop and remove all data (including vectors)
docker-compose down -v

# View logs from all services
docker-compose logs -f

# View logs from specific service
docker-compose logs -f backend
docker-compose logs -f ollama

# Restart a specific service
docker-compose restart backend

# Rebuild after code changes
docker-compose up -d --build
```

### Ollama Management

```bash
# List downloaded models
docker exec rag_ollama ollama list

# Pull a different model
docker exec rag_ollama ollama pull llama3.2:3b

# Remove a model
docker exec rag_ollama ollama rm llama3.1:8b

# Test Ollama directly
docker exec rag_ollama ollama run llama3.1:8b "Hello!"

# Check Ollama API
curl http://localhost:11434/api/tags
```

### System Health & Debugging

```bash
# Check overall system health
curl http://localhost:8000/health

# View system statistics
curl http://localhost:8000/stats

# Check which LLM provider is active
curl http://localhost:8000/ | jq '.llm_provider'

# Debug vector store status
curl http://localhost:8000/debug/qdrant
```

## Dual LLM Architecture

This system supports two LLM providers, configurable via environment variables:

### Groq (Cloud-Based)

**Advantages:**
- Faster responses
- High quality
- Minimal local compute required
- No setup required and Works immediately with API key

**Best for:**
- Production deployments with speed requirements
- CPU only systems (laptops without GPU)
- Applications requiring high quality responses
- Development and quick prototyping

### Ollama (Local)

**Advantages:**
- Privacy focused as all data stays on your infrastructure
- No API costs, unlimited usage
- Works without internet connection
- Full control with choosing models, customize parameters

**Best for:**
- Privacy sensitive applications (legal, medical, financial)
- Offline environments
- Organizations with GPU infrastructure
- Scenarios requiring data sovereignty

### Automatic Fallback Mechanism

The system implements intelligent fallback logic:

1. Attempts to use the configured `LLM_PROVIDER` (Groq or Ollama)
2. Verifies the provider is accessible with a test connection
3. If primary fails, automatically switches to Groq
4. All provider changes are logged for monitoring


### Switching Between Providers

To switch LLM providers, simply update `.env` and restart:

```bash
# Switch to Groq
Set LLM_PROVIDER=groq # edit the .env file
docker-compose restart backend

# Switch to Ollama
Set LLM_PROVIDER=ollama # edit the .env file
docker-compose restart backend
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System information, version, and active LLM provider |
| `/health` | GET | Health check with system statistics |
| `/ingest` | POST | Upload and process a document (multipart/form-data) |
| `/chat` | POST | Query the knowledge base with RAG |
| `/files` | GET | List all uploaded files with metadata |
| `/files/{filename}` | DELETE | Delete a specific file from knowledge base |
| `/files` | DELETE | Clear all files and reset system |
| `/stats` | GET | Detailed system statistics and metrics |
| `/export` | GET | Export complete knowledge base (backup) |
| `/debug/qdrant` | GET | Debug Qdrant vector store status |

For detailed API documentation, visit http://localhost:8000/docs after starting the system.

## Troubleshooting

### Common Issues

#### Backend won't start
```bash
# Check logs for errors
docker-compose logs backend

# Verify GROQ_API_KEY is set
cat .env | grep GROQ_API_KEY

# Rebuild container
docker-compose build --no-cache backend
docker-compose up -d
```

#### Ollama Connection Failed
```bash
# Ensure Ollama service is running
docker-compose ps ollama

# Check Ollama logs
docker-compose logs ollama

# Verify model is downloaded
docker exec rag_ollama ollama list

# Restart services in sequence
docker-compose restart ollama
sleep 10
docker-compose restart backend
```

#### Slow Ollama Performance (CPU)
This is expected behavior on CPU only systems. Options:
- Switch to Groq for faster responses: `LLM_PROVIDER=groq`
- Use a smaller Ollama model: `OLLAMA_MODEL=llama3.2:3b`
- Add GPU support

#### Port Already in Use
```bash
# Check what's using the port (Windows)
netstat -ano | findstr :8000

# Check what's using the port (Mac/Linux)
lsof -i :8000

# Change port in docker-compose.yml if needed
```

## Security Considerations

### Current Implementation (Development/POC)
- API keys stored in `.env` file (not committed to git)
- No authentication on API endpoints
- CORS configured to allow all origins
- No rate limiting implemented

### **Secrets Management**

The current use of a local `.env` file for storing sensitive credentials like `GROQ_API_KEY` is a **security risk** and not suitable for production. For production use a dedicated, mananged secret service like AWS Secret Manager/HashiCorp Vault/similar.

## Production & Scaling Architecture Considerations

This RAG system demonstrates core concepts using local persistence (Qdrant on disk) and an in-memory lexical index (BM25). For a true production deployment supporting large document volumes and concurrent users, the following architectural upgrades are essential:

### Core Enterprise Requirements
- **Authentication & Authorization (AuthN/AuthZ):** Implement robust user verification and session management.
- **Document-Level Access Control:** Ensure query results respect user permissions on the source files.
- **Audit Logging:** Detailed logging of file uploads, deletes, and user queries for compliance and monitoring.
- **Enterprise Connectors:** Integrate with common data sources (SharePoint, Confluence, GitHub) for automated ingestion.

### Scaling & Performance Enhancements

| Component | Current PoC Approach | Production Solution (Scaling) | Benefit |
| :--- | :--- | :--- | :--- |
| **Vector Index** | Local Qdrant instance on the application server disk. | **Managed/Cloud Qdrant** (or other distributed vector DB like Pinecone/Weaviate). | Enables horizontal scaling, high availability, and separation of concerns. |
| **Lexical Index** | In-memory BM25 index of all text chunks (`self.all_texts`). | **Dedicated Search Engine** (e.g., Elasticsearch or OpenSearch). | Eliminates the need to load all document text into the application server's RAM, preventing memory exhaustion. |
| **Hybrid Retrieval** | Ensemble Retriever combining in-memory BM25 and local Qdrant. | Orchestrates calls to the external Vector DB and external Search Engine. | Maintains superior result quality (semantic + keyword) at massive scale. |
| **Data Persistence** | Local JSON files for data backup. | **Cloud Object Storage (S3/GCS)** or a **Managed Relational Database (PostgreSQL)**. | Provides centralized, durable storage for original documents and application metadata, simplifying multi-server deployments. |
| **LLM Infrastructure** | Single Groq/Ollama instance. | **Load-balanced LLM endpoints** with auto-scaling and request queuing. | Handles traffic spikes and ensures high availability for inference requests. |

## Development

### Local Development (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements-frozen.txt

# Set environment variables
export GROQ_API_KEY="your_key"
export LLM_PROVIDER="groq"
export QDRANT_STORAGE_PATH="./qdrant_storage"

# Run backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, serve frontend
cd frontend
python -m http.server 80
```