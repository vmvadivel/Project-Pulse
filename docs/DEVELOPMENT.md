# Development Guide

Guide for running Project Pulse locally without Docker.

---

## Local Development Setup (Windows)

### Prerequisites

- **Python 3.11+**
- **Git**
- **Groq API Key**

### Step 1: Clone Repository

```cmd
git clone https://github.com/vmvadivel/Project-Pulse.git
cd project-pulse
```

### Step 2: Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```cmd
cd backend
pip install -r requirements-frozen.txt
```

**Note:** This installs CPU-only PyTorch. Installation may take 5-10 minutes.

### Step 4: Configure Environment

Create `.env` file in project root:

```env
GROQ_API_KEY="gsk_your_api_key_here"
LLM_PROVIDER=groq

# Ollama configuration
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_BASE_URL=http://ollama:11434 #for docker
# OLLAMA_MODEL=llama3.1:8b

# RAG Enhancement Features
ENABLE_RERANKING=true
ENABLE_CONTEXTUAL_ENRICHMENT=true

# Enable debug endpoints and verbose logging
DEBUG_MODE=true
ENABLE_DEBUG_ENDPOINTS=true
```

### Step 5: Run Backend

```cmd
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: http://localhost:8000

API docs at: http://localhost:8000/docs

### Step 6: Run Frontend

Open new terminal:

```cmd
cd frontend
python -m http.server 80
```

Frontend will be at: http://localhost

---

## Code Architecture

### Backend Modules

**main.py** - Application entry point
- FastAPI app initialization
- API endpoint definitions
- Middleware configuration (CORS, error handlers)
- Request/response handling

**config.py** - Configuration management
- Environment variable loading
- Model initialization (LLM, embeddings, reranker)
- Configuration constants and feature flags

**services.py** - Business logic
- DocumentStore class (manages documents, vectors, QA chain)
- Document processing (loading, chunking, enrichment)
- Vector storage management (Qdrant operations)
- Retrieval and reranking implementation

**utils.py** - Utility functions
- File validation (size, type, uniqueness)
- Format conversion (file size, timestamps)
- Helper functions

**exceptions.py** - Error handling
- Custom exception definitions with error codes
- Error response formatting
- HTTP status code mapping
- Error handler registration

### Frontend

Simple single-page application with vanilla JavaScript (no framework).

---

## Running Tests

Basic tests exist in `backend/tests/`:
- `test_error_integration.py`
- `test_exceptions.py`

**To run tests:**
```cmd
cd backend
pytest tests/
```

---

## Debugging

### Enable Debug Mode

Edit `.env`:
```env
DEBUG_MODE=true
ENABLE_DEBUG_ENDPOINTS=true
```

Restart application.

---

## Common Development Issues

### Import Errors

**Issue:** `ModuleNotFoundError`

**Solution:**
```cmd
# Ensure virtual environment is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements-frozen.txt
```

### Port Already in Use

**Issue:** `Address already in use: 8000`

**Solution:**
```cmd
# Find process
netstat -ano | findstr :8000

# Kill process
taskkill /PID <PID> /F

# Or use different port
uvicorn main:app --reload --port 8001
```

### Qdrant Connection Issues

**Issue:** Cannot connect to Qdrant

**Solution:**
- Qdrant runs in-memory when not using Docker
- Check `QDRANT_STORAGE_PATH` is writable
- Verify `./qdrant_storage/` directory exists

---

## Next Steps

- [Architecture Guide](ARCHITECTURE.md) - Understand the system
- [API Reference](API.md) - Explore endpoints
- [Troubleshooting](TROUBLESHOOTING.md) - Fix issues

---

[← Back to Main README](../README.md)