# Installation Guide

Complete guide for setting up Project Pulse on Windows with Docker.

---

> **⚠️ Tested Environment**  
> This guide is based on testing performed on **Windows 11 + Docker Desktop**.  
> Mac and Linux users may need to adapt commands - community contributions welcome!

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10 (with WSL2) or Windows 11
- **RAM**: 8GB minimum (16GB recommended for Ollama)
- **Disk Space**: 10GB free (for Docker images and Ollama models if used)
- **Docker**: Docker Desktop for Windows v4.0+
- **Internet**: Required for initial setup and Groq API

### For Ollama (Local LLM)
- **RAM**: 16GB+ strongly recommended
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM)
- **Disk Space**: Additional 5-15GB per model
- ⚠️ **CPU-only performance**: Tested but impractical (queries take multiple minutes)

---

## Prerequisites

### 1. Install Docker Desktop

**Download and install Docker Desktop for Windows:**
- https://docs.docker.com/desktop/install/windows-install/

**Requirements:**
- Enable WSL2 backend
- Allocate at least 8GB RAM to Docker (16GB if using Ollama)
- Enable virtualization in BIOS

**Verify installation:**
```cmd
docker --version
docker-compose --version
```

Expected output:
```
Docker version 24.0.x
Docker Compose version v2.x.x
```

### 2. Get Groq API Key

1. Visit https://console.groq.com
2. Sign up for a free account
3. Navigate to "API Keys" section
4. Create a new API key
5. Copy the key (starts with `gsk_...`)

**Note**: Groq's free tier is sufficient for development and testing.

---

## Installation Steps

### Step 1: Clone the Repository

```cmd
git clone <repository-url>
cd project-pulse
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root directory:

#### Option A: Using Groq (Recommended for CPU-only systems)

```env
# LLM Configuration
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
LLM_PROVIDER=groq

# Optional: Advanced RAG Configuration
ENABLE_RERANKING=true
ENABLE_CONTEXTUAL_ENRICHMENT=true

# Optional: Debug Mode
DEBUG_MODE=false
ENABLE_DEBUG_ENDPOINTS=false
```

#### Option B: Using Ollama (For local/private deployment with GPU)

```env
# LLM Configuration
GROQ_API_KEY=gsk_your_actual_groq_api_key_here  # Still required as fallback
LLM_PROVIDER=ollama

# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.1:8b

# Optional: Advanced RAG Configuration
ENABLE_RERANKING=true
ENABLE_CONTEXTUAL_ENRICHMENT=true

# Optional: Debug Mode
DEBUG_MODE=false
ENABLE_DEBUG_ENDPOINTS=false
```

**Important Notes:**
- `GROQ_API_KEY` is always required (used as fallback even with Ollama)
- Never commit `.env` file to version control (already in `.gitignore`)
- For production use, consider proper secrets management

### Step 3: Start the Application

```cmd
# Start all services in detached mode
docker-compose up -d
```

**Expected output:**
```
[+] Running 3/3
 ✔ Container rag_ollama     Started
 ✔ Container rag_backend    Started
 ✔ Container rag_frontend   Started
```

**Check service status:**
```cmd
docker-compose ps
```

Expected status:
```
NAME            STATUS                   PORTS
rag_backend     Up (healthy)            0.0.0.0:8000->8000/tcp
rag_frontend    Up                      0.0.0.0:80->80/tcp
rag_ollama      Up                      0.0.0.0:11434->11434/tcp
```

### Step 4: Pull Ollama Model (If Using Ollama)

⚠️ **Skip this step if using `LLM_PROVIDER=groq`**

**Download the Llama 3.1 8B model (~4.7GB, takes 3-5 minutes):**
```cmd
docker exec rag_ollama ollama pull llama3.1:8b
```

**Verify model is available:**
```cmd
docker exec rag_ollama ollama list
```

Expected output:
```
NAME              ID              SIZE      MODIFIED
llama3.1:8b       42182419e950    4.7 GB    2 minutes ago
```

**Restart backend to establish connection:**
```cmd
docker-compose restart backend
```

### Step 5: Verify Installation

#### Check Backend Health
```cmd
curl http://localhost:8000/health
```

Expected response (abbreviated):
```json
{
  "status": "healthy",
  "stats": {
    "total_files": 0,
    ...
  }
}
```

#### Check Active LLM Provider
```cmd
curl http://localhost:8000/
```

Look for `"llm_provider"` field - should show `"groq"` or `"ollama"`

#### Access Frontend
Open browser and navigate to: **http://localhost**

You should see the Project Pulse interface.

---

## LLM Provider Configuration

### Groq (Cloud-Based)

**Characteristics:**
- Fast inference (800+ tokens/sec)
- Works well on any system (no GPU needed)
- Requires internet connection
- Free tier available

**Best for:**
- Development and testing
- CPU only systems
- Quick prototyping
- Systems without GPU

**Configuration:**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

### Ollama (Local)

**Characteristics:**
- Complete privacy (all processing local)
- No API costs
- Works offline
- **CPU-only performance**: Tested but impractical (queries take multiple minutes)
- **With GPU**: Acceptable performance for production use

**Best for:**
- Privacy sensitive applications
- Organizations with GPU infrastructure
- Offline environments
- Users with NVIDIA GPU (8GB+ VRAM recommended)

**Configuration:**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.1:8b
GROQ_API_KEY=gsk_...  # Fallback
```

**Performance Note:** Tested on CPU-only system - queries took multiple minutes to complete. Only practical with dedicated GPU.

### Switching Between Providers

To switch LLM providers:

```cmd
# 1. Edit .env file
# Change: LLM_PROVIDER=groq
# To:     LLM_PROVIDER=ollama

# 2. Restart backend (preserves data)
docker-compose restart backend
```

---

## Alternative Ollama Models

### Smaller Model (Faster on CPU, Less Accurate)
```cmd
# Pull Llama 3.2 3B (~2GB)
docker exec rag_ollama ollama pull llama3.2:3b
```

Update `.env`:
```env
OLLAMA_MODEL=llama3.2:3b
```

**Note:** Still slow on CPU, but faster than 8B model.

### Larger Model (Better Quality, Requires GPU)
```cmd
# Pull Llama 3.1 70B (~40GB - requires significant GPU memory)
docker exec rag_ollama ollama pull llama3.1:70b
```

Update `.env`:
```env
OLLAMA_MODEL=llama3.1:70b
```

**Requirements:** NVIDIA GPU with 40GB+ VRAM (or distributed across multiple GPUs)

**Restart backend after model change:**
```cmd
docker-compose restart backend
```

---

## GPU Support (Optional)

If you have an NVIDIA GPU and want to use Ollama with GPU acceleration:

### Prerequisites
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on host Windows
- Docker Desktop configured for GPU pass-through

### Enable GPU in Docker Compose

Edit `docker-compose.yml`:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: rag_ollama
    # Add GPU configuration
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # ... rest of configuration
```

### Verify GPU Access

```cmd
# Check GPU is available in container
docker exec rag_ollama nvidia-smi
```

**Note:** GPU configuration has not been personally tested but follows standard Docker GPU setup patterns.

---

## Post-Installation Configuration

### Adjusting Docker Resources

If experiencing performance issues:

**Docker Desktop → Settings → Resources:**
- **Memory**: Increase to 8GB+ (16GB for Ollama)
- **CPU**: Allocate 4+ cores if available
- **Disk**: Ensure sufficient space for images and models

### Enabling Debug Mode

For development or troubleshooting:

Edit `.env`:
```env
DEBUG_MODE=true
ENABLE_DEBUG_ENDPOINTS=true
```

Restart:
```cmd
docker-compose restart backend
```

Access debug endpoint:
```cmd
curl http://localhost:8000/debug/qdrant
```

### Advanced RAG Configuration

Configure retrieval behavior in `.env`:

```env
# Reranking
ENABLE_RERANKING=true
RETRIEVAL_K_BEFORE_RERANK=20
RETRIEVAL_K_AFTER_RERANK=7
MIN_RELEVANCE_SCORE=0.3

# Contextual Enrichment
ENABLE_CONTEXTUAL_ENRICHMENT=true
CONTEXT_WINDOW_CHARS=200
```

---

## Troubleshooting Installation Issues

### Backend Won't Start

**Check logs:**
```cmd
docker-compose logs backend
```

**Common causes:**
- Missing `GROQ_API_KEY` in `.env`
- Invalid API key format
- Port 8000 already in use

**Solution:** See [Troubleshooting Guide](TROUBLESHOOTING.md#backend-issues)

### Ollama Connection Failed

**Check Ollama status:**
```cmd
docker-compose ps ollama
docker-compose logs ollama
```

**Verify Ollama is responding:**
```cmd
curl http://localhost:11434/api/tags
```

**Solution:** See [Troubleshooting Guide](TROUBLESHOOTING.md#ollama-issues)

### Port Already in Use

**Check what's using port 8000 (Windows):**
```cmd
netstat -ano | findstr :8000
```

**Solutions:**
- Stop the conflicting process
- Or change port in `docker-compose.yml` (e.g., 8001:8000)

**Full troubleshooting:** See [Troubleshooting Guide](TROUBLESHOOTING.md)

---

## Uninstallation

### Remove Containers Only (Keep Data)
```cmd
docker-compose down
```

### Remove Everything (Including Data)
```cmd
# Stop and remove containers, networks, volumes
docker-compose down -v

# Remove Ollama models
docker volume rm project-pulse_ollama_data

# Remove local persistent storage
rmdir /s backend\qdrant_storage
rmdir /s backend\temp
```

---

## Next Steps

- [Architecture Guide](ARCHITECTURE.md) - Understand how it works
- [API Reference](API.md) - Explore available endpoints (or visit `/docs` when running)
- [Troubleshooting](TROUBLESHOOTING.md) - Fix common issues
- [Docker Reference](DOCKER.md) - Container management commands

---

[← Back to Main README](../README.md)