# Setup Guide

How to get Project Pulse running on your machine. I've only tested this on Windows 11 with Docker Desktop, but it should work on other platforms too.

## What You Need

- Docker Desktop installed
- A Groq API key (free at https://console.groq.com)
- 8GB RAM minimum (16GB if you want to use Ollama)

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/vmvadivel/Project-Pulse.git
cd Project-Pulse
```

**2. Create a .env file**

In the project root, create a file called `.env`:

```env
GROQ_API_KEY=gsk_your_actual_key_here
LLM_PROVIDER=groq

# optional: enable advanced features
ENABLE_RERANKING=true
ENABLE_CONTEXTUAL_ENRICHMENT=true

# optional: debug mode
DEBUG_MODE=false
```

**3. Start it up**
```bash
docker-compose up -d
```

**4. Check if it's running**
```bash
docker-compose ps
```

You should see three containers running:
- rag_backend
- rag_frontend  
- rag_ollama (even if you're using Groq)

**5. Access it**
- Frontend: http://localhost
- API docs: http://localhost:8000/docs

That's it!

## Using Ollama Instead of Groq

If you want to run everything locally (for privacy or offline use):

**1. Edit .env**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.1:8b
GROQ_API_KEY=gsk_still_need_this_as_fallback
```

**2. Pull the model**
```bash
docker exec rag_ollama ollama pull llama3.1:8b
```

This downloads about 4.7GB and takes a few minutes.

**3. Restart the backend**
```bash
docker-compose restart backend
```

**Note:** I tested Ollama on CPU and it was painfully slow (3+ minutes per query). You really need a GPU for decent performance. If you don't have a GPU, stick with Groq.

## Common Issues

### Backend won't start

Check the logs:
```bash
docker-compose logs backend
```

Usually it's because:
- Missing GROQ_API_KEY in .env
- Invalid API key format
- Port 8000 already in use

### Port already in use

Find what's using it:
```bash
# Windows
netstat -ano | findstr :8000

# Kill it
taskkill /PID <PID> /F
```

Or change the port in docker-compose.yml

### Ollama is slow

Yeah, it's slow on CPU. Switch back to Groq:
```env
LLM_PROVIDER=groq
```

### Can't access frontend

Make sure nothing else is using port 80. Or change it in docker-compose.yml:
```yaml
ports:
  - "8080:80"  # use port 8080 instead
```

## Running Without Docker

If you prefer not to use Docker:

```bash
# create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# install dependencies
cd backend
pip install -r requirements-frozen.txt

# set environment variables
set GROQ_API_KEY=your_key  # Windows

# run backend
uvicorn main:app --reload
```

Then open `frontend/index.html` in your browser.

## Configuration Options

You can tweak these in .env:

**Reranking:**
```env
ENABLE_RERANKING=true
RETRIEVAL_K_BEFORE_RERANK=20
RETRIEVAL_K_AFTER_RERANK=7
MIN_RELEVANCE_SCORE=0.3
```

**Contextual enrichment:**
```env
ENABLE_CONTEXTUAL_ENRICHMENT=true
CONTEXT_WINDOW_CHARS=200
```

**Debug mode:**
```env
DEBUG_MODE=true
ENABLE_DEBUG_ENDPOINTS=true
```

Debug mode gives you access to http://localhost:8000/debug/qdrant for checking vector store status.

## Updating

Pull latest changes:
```bash
git pull
docker-compose up -d --build
```

## Uninstalling

Remove everything:
```bash
# stop and remove containers
docker-compose down -v

# remove data directories
rm -rf backend/qdrant_storage
rm -rf backend/temp
```

## Next Steps

- Upload some documents and try it out
- Check out the API docs at /docs
- Read [ARCHITECTURE.md](ARCHITECTURE.md) if you want to understand how it works

## Getting Help

If you run into issues:
1. Check the logs: `docker-compose logs backend`
2. Make sure Docker Desktop is running
3. Verify your .env file has the right format
4. Try rebuilding: `docker-compose up -d --build`

Still stuck? Open an issue on GitHub with the error message and I'll try to help.