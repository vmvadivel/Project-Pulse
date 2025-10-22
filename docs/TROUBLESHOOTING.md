# Troubleshooting Guide

Common issues and solutions for Project Pulse on Windows.

---

> **⚠️ Tested Environment**  
> This guide is based on troubleshooting performed on **Windows 11 + Docker Desktop**.  
> If you're using a different setup, some solutions may need adaptation.

---

## Table of Contents

- [Docker Issues](#docker-issues)
- [Backend Issues](#backend-issues)
- [Ollama Issues](#ollama-issues)
- [Performance Issues](#performance-issues)
- [File Upload Issues](#file-upload-issues)
- [Query Issues](#query-issues)
- [Network Issues](#network-issues)
- [Getting Help](#getting-help)

---

## Docker Issues

### Issue: "Cannot connect to Docker daemon"

**Symptoms:**
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution:**

1. **Start Docker Desktop**
   - Look for Docker whale icon in system tray
   - If not running, launch Docker Desktop from Start menu
   - Wait for "Docker Desktop is running" message

2. **Verify Docker is running:**
   ```cmd
   docker --version
   docker ps
   ```

3. **If Docker Desktop won't start:**
   - Check if Hyper-V or WSL2 is enabled
   - Restart your computer
   - Reinstall Docker Desktop if needed

---

### Issue: "Port already in use"

**Symptoms:**
```
Error: Bind for 0.0.0.0:8000 failed: port is already allocated
```

**Solution:**

**Find what's using the port:**
```cmd
netstat -ano | findstr :8000
```

**Kill the process:**
```cmd
taskkill /PID <PID_NUMBER> /F
```

**Or change the port in `docker-compose.yml`:**
```yaml
services:
  backend:
    ports:
      - "8001:8000"  # Change external port to 8001
```

Then access at: http://localhost:8001

---

### Issue: "docker-compose: command not found"

**Symptoms:**
```
'docker-compose' is not recognized as an internal or external command
```

**Solution:**

**Use Docker Compose V2 syntax:**
```cmd
docker compose up -d
```

(Note: `docker compose` instead of `docker-compose`)

**Or install Docker Compose V1:**
- Download from: https://github.com/docker/compose/releases
- Place in a folder in your PATH

---

### Issue: "Container exits immediately"

**Symptoms:**
```cmd
docker-compose ps

NAME            STATUS
rag_backend     Exited (1)
```

**Solution:**

**1. Check logs for errors:**
```cmd
docker-compose logs backend
```

**2. Common causes:**
- Missing `GROQ_API_KEY` in `.env` file
- Invalid API key format
- Python dependency issues

**3. Verify `.env` file:**
```cmd
type .env
```

Make sure it contains:
```env
GROQ_API_KEY=gsk_your_actual_key_here
LLM_PROVIDER=groq
```

**4. Rebuild container:**
```cmd
docker-compose down
docker-compose build --no-cache backend
docker-compose up -d
```

---

### Issue: "Out of memory"

**Symptoms:**
- Containers crash randomly
- Docker Desktop shows high memory usage
- Windows becomes slow

**Solution:**

**Increase Docker memory allocation:**

1. Docker Desktop → Settings → Resources
2. Increase **Memory** to 8GB+ (16GB for Ollama)
3. Increase **CPU** to 4+ cores if available
4. Click "Apply & Restart"

**Check current usage:**
```cmd
docker stats
```

---

## Backend Issues

### Issue: "Application startup failed"

**Symptoms:**
```
Error: MissingAPIKey - GROQ_API_KEY not found
```

**Solution:**

**1. Check `.env` file exists:**
```cmd
dir .env
```

**2. Verify content:**
```cmd
type .env
```

**3. Ensure correct format (no spaces around `=`):**
```env
# Wrong
GROQ_API_KEY = gsk_...

# Correct
GROQ_API_KEY=gsk_...
```

**4. Restart backend:**
```cmd
docker-compose restart backend
```

---

### Issue: "Health check failing"

**Symptoms:**
```cmd
docker-compose ps

NAME            STATUS
rag_backend     Up (unhealthy)
```

**Solution:**

**1. Check backend logs:**
```cmd
docker-compose logs backend --tail 50
```

**2. Test health endpoint manually:**
```cmd
curl http://localhost:8000/health
```

**3. If connection refused:**
- Backend might still be starting (wait 30-60 seconds)
- Check if port 8000 is accessible
- Check if backend container is actually running: `docker-compose ps`

**4. Restart backend:**
```cmd
docker-compose restart backend
```

---

### Issue: "Import errors / Module not found"

**Symptoms:**
```
ModuleNotFoundError: No module named 'langchain'
```

**Solution:**

**Rebuild container with fresh dependencies:**
```cmd
docker-compose build --no-cache backend
docker-compose up -d
```

**Verify packages installed:**
```cmd
docker-compose exec backend pip list | findstr langchain
```

---

## Ollama Issues

### Issue: "Ollama connection failed"

**Symptoms:**
```
Failed to connect to Ollama at http://ollama:11434. Falling back to Groq.
```

**Solution:**

**1. Check Ollama container status:**
```cmd
docker-compose ps ollama
```

**2. Check Ollama logs:**
```cmd
docker-compose logs ollama
```

**3. Verify Ollama is responding:**
```cmd
curl http://localhost:11434/api/tags
```

**4. Restart Ollama:**
```cmd
docker-compose restart ollama
timeout /t 10
docker-compose restart backend
```

---

### Issue: "Model not found"

**Symptoms:**
```
Error: model 'llama3.1:8b' not found
```

**Solution:**

**1. List available models:**
```cmd
docker exec rag_ollama ollama list
```

**2. Pull the model:**
```cmd
docker exec rag_ollama ollama pull llama3.1:8b
```

This downloads ~4.7GB and takes 3-5 minutes.

**3. Verify model is available:**
```cmd
docker exec rag_ollama ollama list
```

**4. Restart backend:**
```cmd
docker-compose restart backend
```

---

### Issue: "Ollama extremely slow"

**Symptoms:**
- Queries take multiple minutes to complete
- High CPU usage (100%)
- Unusable response times

**Root Cause:** Running on CPU without GPU acceleration

**Solution:**

**Option 1: Switch to Groq (Recommended)**

Edit `.env`:
```env
LLM_PROVIDER=groq
```

Restart:
```cmd
docker-compose restart backend
```

**Option 2: Use smaller model (still slow, but faster)**

Pull smaller model:
```cmd
docker exec rag_ollama ollama pull llama3.2:3b
```

Update `.env`:
```env
OLLAMA_MODEL=llama3.2:3b
```

Restart:
```cmd
docker-compose restart backend
```

**Note:** Ollama on CPU-only systems results in queries taking multiple minutes. Only practical with dedicated GPU. For acceptable performance, use Groq.

---

## Performance Issues

### Issue: "Slow query responses"

**Symptoms:**
- Queries take a long time to return results
- High CPU usage during queries

**Solution:**

**If using Ollama on CPU:**
- Ollama queries take multiple minutes on CPU-only systems
- **Switch to Groq for acceptable performance:**
  ```env
  LLM_PROVIDER=groq
  ```

**If using Groq and still slow:**
- Check internet connection
- Verify Groq API is accessible: https://console.groq.com
- Check for rate limiting (see error messages)

**To reduce response time:**

Edit `backend/config.py`:
```python
RETRIEVAL_K_AFTER_RERANK = 5  # Reduced from 7
```

Or disable reranking:
```env
ENABLE_RERANKING=false
```

---

### Issue: "Slow document upload"

**Symptoms:**
- Upload takes 10+ seconds for small files
- High CPU usage during upload

**Solutions:**

**1. Check file size:**
```cmd
dir document.pdf
```

**2. Reduce document complexity:**
- Compress PDF if very large
- Split into smaller files

**3. Monitor during upload:**
```cmd
docker stats rag_backend
```

**4. Increase Docker resources:**
- Docker Desktop → Settings → Resources
- Increase CPU allocation to 4+ cores
- Increase Memory to 8GB+

---

### Issue: "High memory usage"

**Symptoms:**
- Backend container using 4+ GB RAM
- System slowdown
- Docker Desktop memory warnings

**Cause:** Large number of documents loaded into BM25 index

**Solutions:**

**1. Check document count:**
```cmd
curl http://localhost:8000/stats
```

**2. If many documents (1000+ chunks):**

Clear old documents:
```cmd
curl -X DELETE http://localhost:8000/files
```

**3. Increase Docker memory:**
- Docker Desktop → Settings → Resources
- Increase Memory to 16GB

---

## File Upload Issues

### Issue: "File too large"

**Symptoms:**
```json
{
  "error": {
    "code": "FILE_002",
    "message": "File size exceeds maximum allowed"
  }
}
```

**Solution:**

**1. Check file size:**
```cmd
dir document.pdf
```

**2. Maximum size is 100MB. Options:**

**Compress PDF:**
- Use online tools (e.g., iLovePDF, SmallPDF)
- Or Adobe Acrobat compression

**Split large files:**
- Break into smaller parts
- Upload separately

**Increase limit (not recommended):**

Edit `backend/config.py`:
```python
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
```

---

### Issue: "Unsupported file type"

**Symptoms:**
```json
{
  "error": {
    "code": "FILE_001",
    "message": "File type not supported"
  }
}
```

**Supported formats:**
- Documents: PDF, DOC, DOCX, TXT, MD
- Spreadsheets: XLS, XLSX, CSV
- Presentations: PPT, PPTX
- Web: HTML, HTM
- Data: JSON, XML

**Solution:**

**Convert to supported format:**
- Save as PDF (most reliable)
- Or save as TXT/DOCX

---

### Issue: "File already exists"

**Symptoms:**
```json
{
  "error": {
    "code": "FILE_005",
    "message": "File already exists in knowledge base"
  }
}
```

**Solution:**

**Option 1: Delete existing file first**
```cmd
curl -X DELETE http://localhost:8000/files/document.pdf
```

**Option 2: Rename the file**
```cmd
ren document.pdf document_v2.pdf
```

---

### Issue: "File corrupted"

**Symptoms:**
```json
{
  "error": {
    "code": "FILE_003",
    "message": "Unable to read file contents"
  }
}
```

**Common Causes:**
- Password-protected PDFs
- Corrupted files
- Scanned images without OCR
- Incompatible file encoding

**Solution:**

**1. Verify file can be opened:**
- Try opening in Adobe Reader or Word
- Check for corruption

**2. For password-protected PDFs:**
- Remove password using Adobe Acrobat or online tools
- Save as unprotected PDF

**3. For scanned documents:**
- Use OCR software to make text searchable
- Or save as image-based PDF with text layer

**4. Try re-saving the file:**
- Open in original application
- Save as new file with different name

---

## Query Issues

### Issue: "No documents available"

**Symptoms:**
```json
{
  "error": {
    "code": "RETRIEVAL_001",
    "message": "No documents available to search"
  }
}
```

**Solution:**

**1. Check if any files are uploaded:**
```cmd
curl http://localhost:8000/stats
```

**2. Upload a document:**
```cmd
curl -X POST http://localhost:8000/ingest -F "file=@document.pdf"
```

---

### Issue: "Poor answer quality"

**Symptoms:**
- Answers don't match document content
- Hallucinated information
- Missing relevant information

**Solutions:**

**1. Enable reranking (if disabled):**
```env
ENABLE_RERANKING=true
```

**2. Enable contextual enrichment:**
```env
ENABLE_CONTEXTUAL_ENRICHMENT=true
```

**3. Improve query specificity:**
```bash
# ❌ Bad: "Tell me about health"
# ✅ Good: "What was my vitamin D level in the September 2024 blood test?"
```

**4. Check which documents were used:**
```cmd
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"your question\"}"
```

Look at `source_files` in the response.

---

### Issue: "Rate limit exceeded"

**Symptoms:**
```json
{
  "error": {
    "code": "LLM_002",
    "message": "AI service rate limit exceeded",
    "details": {
      "retry_after": 60
    }
  }
}
```

**Cause:** Exceeded Groq free tier limits

**Solutions:**

**Option 1: Wait and retry**
```cmd
timeout /t 60
# Retry your query
```

**Option 2: Switch to Ollama (if you have GPU)**
```env
LLM_PROVIDER=ollama
```

**Note:** Ollama on CPU will be extremely slow (multiple minutes per query).

---

## Network Issues

### Issue: "Cannot access frontend"

**Symptoms:**
- http://localhost doesn't load
- Browser shows "Connection refused"
- Page not found error

**Solution:**

**1. Check frontend container:**
```cmd
docker-compose ps frontend
```

**2. Check if port 80 is free:**
```cmd
netstat -ano | findstr :80
```

**3. If port 80 is in use, change port:**

Edit `docker-compose.yml`:
```yaml
frontend:
  ports:
    - "8080:80"  # Change to 8080
```

Restart:
```cmd
docker-compose up -d
```

Access at: http://localhost:8080

---

### Issue: "CORS errors in browser"

**Symptoms:**
```
Access to fetch at 'http://localhost:8000/chat' from origin 'http://localhost'
has been blocked by CORS policy
```

**Solution:**

**1. Verify CORS configuration:**

Your `backend/main.py` should have:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**2. Access via proper URL:**
- Use http://localhost (not http://127.0.0.1)
- Don't mix protocols (http/https)

**3. Restart backend:**
```cmd
docker-compose restart backend
```

---

## Getting Help

### Collect Diagnostic Information

```cmd
# System info
docker --version
docker-compose --version

# Container status
docker-compose ps

# Recent logs
docker-compose logs --tail=50 backend
docker-compose logs --tail=50 ollama

# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/stats
```

### Enable Debug Mode

Edit `.env`:
```env
DEBUG_MODE=true
ENABLE_DEBUG_ENDPOINTS=true
```

Restart:
```cmd
docker-compose restart backend
```

Access debug info:
```cmd
curl http://localhost:8000/debug/qdrant
```

### Report an Issue

If you need to report an issue, include:
1. Error message (full text)
2. Steps to reproduce
3. System info (Windows version, Docker version)
4. Relevant logs (from commands above)
5. `.env` configuration (redact API keys!)

---

## Next Steps

- [Installation Guide](INSTALLATION.md) - Setup instructions
- [Architecture Guide](ARCHITECTURE.md) - System design
- [API Reference](API.md) - Endpoint documentation
- [Docker Reference](DOCKER.md) - Container management

---

[← Back to Main README](../README.md)