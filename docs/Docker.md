# Docker Reference

Docker commands and container management for Project Pulse on Windows.

---

> **⚠️ Tested Environment**  
> Commands tested on **Windows 11 + Docker Desktop**.

---

## Table of Contents

- [Basic Commands](#basic-commands)
- [Service Management](#service-management)
- [Ollama Management](#ollama-management)
- [Logs and Monitoring](#logs-and-monitoring)
- [Data and Volumes](#data-and-volumes)
- [Maintenance](#maintenance)
- [GPU Support](#gpu-support-optional)

---

## Basic Commands

### Starting and Stopping

```cmd
REM Start all services (detached mode)
docker-compose up -d

REM Start all services (foreground, see logs)
docker-compose up

REM Start specific service
docker-compose up -d backend

REM Stop all services (keeps containers)
docker-compose stop

REM Stop specific service
docker-compose stop backend

REM Stop and remove all containers
docker-compose down

REM Stop and remove containers + volumes (deletes data)
docker-compose down -v
```

### Restarting Services

```cmd
REM Restart all services
docker-compose restart

REM Restart specific service
docker-compose restart backend

REM Restart with rebuild
docker-compose up -d --build

REM Force recreate containers (reload .env changes)
docker-compose up -d --force-recreate
```

### Building Images

```cmd
REM Build all images
docker-compose build

REM Build specific service
docker-compose build backend

REM Build without cache (clean build)
docker-compose build --no-cache

REM Build and start
docker-compose up -d --build
```

---

## Service Management

### Checking Status

```cmd
REM List all containers
docker-compose ps

REM Detailed status
docker-compose ps -a

REM Check health status
docker inspect rag_backend --format="{{.State.Health.Status}}"

REM List running containers
docker ps

REM List all containers including stopped
docker ps -a
```

### Executing Commands in Containers

```cmd
REM Open shell in backend container
docker-compose exec backend bash

REM Run a command in backend
docker-compose exec backend python --version

REM Check environment variables
docker-compose exec backend env

REM Check Python packages
docker-compose exec backend pip list
```

### Resource Usage

```cmd
REM View resource usage (live)
docker stats

REM View specific container
docker stats rag_backend

REM One-time snapshot
docker stats --no-stream
```

---

## Ollama Management

### Model Operations

```cmd
REM List downloaded models
docker exec rag_ollama ollama list

REM Pull a model
docker exec rag_ollama ollama pull llama3.1:8b
docker exec rag_ollama ollama pull llama3.2:3b
docker exec rag_ollama ollama pull llama3.1:70b

REM Remove a model
docker exec rag_ollama ollama rm llama3.1:8b

REM Show model information
docker exec rag_ollama ollama show llama3.1:8b
```

### Testing Ollama

```cmd
REM Test Ollama directly
docker exec rag_ollama ollama run llama3.1:8b "Hello, how are you?"

REM Check Ollama API
curl http://localhost:11434/api/tags

REM Check Ollama version
docker exec rag_ollama ollama --version

REM Test model generation
curl http://localhost:11434/api/generate -d "{\"model\": \"llama3.1:8b\", \"prompt\": \"Why is the sky blue?\", \"stream\": false}"
```

---

## Logs and Monitoring

### Viewing Logs

```cmd
REM View logs from all services
docker-compose logs

REM View logs from specific service
docker-compose logs backend
docker-compose logs frontend
docker-compose logs ollama

REM Follow logs in real-time
docker-compose logs -f backend

REM View last N lines
docker-compose logs --tail=50 backend

REM View logs with timestamps
docker-compose logs -t backend

REM Save logs to file
docker-compose logs backend > backend_logs.txt
```

### Filtering Logs

```cmd
REM Filter by log level
docker-compose logs backend | findstr ERROR
docker-compose logs backend | findstr WARNING
docker-compose logs backend | findstr INFO

REM Filter by keyword
docker-compose logs backend | findstr "Ingestion"
docker-compose logs backend | findstr "query"

REM Count errors
docker-compose logs backend | findstr /C:ERROR
```

---

## Data and Volumes

### Volume Management

```cmd
REM List volumes
docker volume ls

REM Inspect volume
docker volume inspect project-pulse_ollama_data

REM Show volume size
docker system df -v

REM Remove specific volume (deletes data)
docker volume rm project-pulse_ollama_data

REM Remove all unused volumes
docker volume prune
```

### Persistent Data Locations

**Qdrant Storage (Windows):**
```cmd
REM Location: .\backend\qdrant_storage\
dir backend\qdrant_storage

REM Simple backup (copy folder)
xcopy /E /I backend\qdrant_storage backup\qdrant_storage_%date%

REM Restore (copy back)
xcopy /E /I backup\qdrant_storage_20251021 backend\qdrant_storage
```

**Temporary Uploads:**
```cmd
REM Location: .\backend\temp\
dir backend\temp

REM Clean temporary files
del /Q backend\temp\*
```

### Data Export via API

```cmd
REM Export knowledge base
curl http://localhost:8000/export > knowledge_base_backup.json

REM Verify export
type knowledge_base_backup.json
```

---

## Maintenance

### Container Updates

```cmd
REM Pull latest images
docker-compose pull

REM Rebuild and restart with latest code
docker-compose up -d --build

REM Update specific service
docker-compose pull backend
docker-compose up -d backend
```

---

## GPU Support (Optional)

### Enable GPU for Ollama

If you have an NVIDIA GPU and want GPU acceleration for Ollama:

**Edit `docker-compose.yml`:**

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

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on Windows host
- Docker Desktop configured for GPU pass-through

**Verify GPU access:**
```cmd
docker exec rag_ollama nvidia-smi
```

**Note:** GPU configuration has not been personally tested but follows standard Docker GPU patterns.

---

## Common Issues

### Container Won't Start

```cmd
REM Check logs for errors
docker-compose logs backend

REM Inspect container
docker inspect rag_backend

REM Check exit code
docker inspect rag_backend --format="{{.State.ExitCode}}"

REM Try starting in foreground
docker-compose up backend
```

### Port Already in Use

```cmd
REM Find process using port
netstat -ano | findstr :8000

REM Kill process
taskkill /PID <PID> /F

REM Or change port in docker-compose.yml
```

### Out of Memory

```cmd
REM Check memory usage
docker stats --no-stream

REM Increase Docker memory limit
REM Docker Desktop → Settings → Resources → Memory
```

---

## Quick Reference

### Most Used Commands

```cmd
REM Start
docker-compose up -d

REM Stop
docker-compose down

REM Restart
docker-compose restart

REM Logs
docker-compose logs -f backend

REM Shell
docker-compose exec backend bash

REM Rebuild
docker-compose up -d --build

REM Clean
docker system prune -a

REM Stats
docker stats

REM Pull Ollama model
docker exec rag_ollama ollama pull llama3.1:8b
```

---

## Next Steps

- [Installation Guide](INSTALLATION.md) - Initial setup
- [Architecture Guide](ARCHITECTURE.md) - System design
- [Troubleshooting](TROUBLESHOOTING.md) - Fix issues

---

[← Back to Main README](../README.md)