# API Reference

Quick reference for Project Pulse REST API endpoints.

---

> ** Complete Interactive Documentation**  
> When the application is running, visit **http://localhost:8000/docs** for:
> - Complete API documentation with request/response schemas
> - Interactive testing (try endpoints directly in browser)
> - Automatic code generation for multiple languages
> - Real-time validation and error messages

This document provides a quick reference only. For detailed information, use the Swagger UI.

---

## Base URL

```
http://localhost:8000
```

---

## Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System information and version |
| `/health` | GET | Health check with statistics |
| `/stats` | GET | Detailed system statistics |
| `/ingest` | POST | Upload and process a document |
| `/chat` | POST | Query the knowledge base |
| `/files` | GET | List all uploaded files |
| `/files/{filename}` | DELETE | Delete specific file |
| `/files` | DELETE | Clear all files |
| `/export` | GET | Export knowledge base |
| `/debug/qdrant` | GET | Debug vector store (if debug enabled) |

---

## Example Usage

### 1. Check System Health

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-21T12:34:56.789012",
  "stats": {
    "total_files": 5,
    "total_chunks": 127,
    ...
  }
}
```

---

### 2. Upload a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "message": "File 'document.pdf' ingested successfully",
  "num_chunks": 45,
  "total_files": 6,
  "processing_time": "5.67s",
  ...
}
```

**Supported Formats:**
- Documents: PDF, DOC, DOCX, TXT, MD
- Spreadsheets: XLS, XLSX, CSV
- Presentations: PPT, PPTX
- Web: HTML, HTM
- Data: JSON, XML

**File Size Limit:** 100MB

---

### 3. Query the System

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What was my vitamin D level?"}'
```

**Response:**
```json
{
  "response": "According to your blood test results...",
  "source_files": ["blood_test_2024.pdf"],
  "num_sources": 3,
  "source_details": [
    {
      "file": "blood_test_2024.pdf",
      "chunk_position": "3/15",
      "rerank_score": 0.89,
      "content_preview": "Your vitamin D level is..."
    }
  ],
  "response_time": "1.23s"
}
```

---

## Error Responses

All errors follow a consistent format with unique error codes.

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

---

## Error Codes Reference

### Configuration Errors (5xx)
| Code | Error | HTTP Status | Description |
|------|-------|-------------|-------------|
| CONFIG_001 | MissingAPIKey | 500 | Required API key not found in environment |
| CONFIG_002 | InvalidAPIKey | 500 | API key rejected by service |
| CONFIG_003 | InvalidConfiguration | 500 | System configuration invalid |

### File Errors (4xx)
| Code | Error | HTTP Status | Description |
|------|-------|-------------|-------------|
| FILE_001 | UnsupportedFileType | 400 | File type not supported for processing |
| FILE_002 | FileTooLarge | 413 | File exceeds 100MB size limit |
| FILE_003 | FileCorrupted | 422 | Unable to read file contents |
| FILE_004 | FileProcessingError | 500 | Error occurred during file processing |
| FILE_005 | FileAlreadyExists | 400 | File already exists in knowledge base |

### Vector Store Errors (5xx)
| Code | Error | HTTP Status | Description |
|------|-------|-------------|-------------|
| VSTORE_001 | QdrantConnectionFailed | 503 | Connection to Qdrant vector database failed |
| VSTORE_002 | QdrantUpsertFailed | 500 | Failed to store documents in vector database |
| VSTORE_003 | CollectionNotFound | 500 | Qdrant collection not found |
| VSTORE_004 | VectorStoreSyncError | 500 | Vector database synchronization failed |

### LLM Errors (4xx/5xx)
| Code | Error | HTTP Status | Description |
|------|-------|-------------|-------------|
| LLM_001 | LLMServiceUnavailable | 503 | LLM service temporarily unavailable |
| LLM_002 | LLMRateLimitExceeded | 429 | API rate limit exceeded |
| LLM_003 | LLMInvalidResponse | 500 | Invalid response received from LLM |
| LLM_004 | LLMTimeout | 504 | LLM request timed out |

### Retrieval Errors (4xx/5xx)
| Code | Error | HTTP Status | Description |
|------|-------|-------------|-------------|
| RETRIEVAL_001 | NoDocumentsIngested | 400 | No documents available in knowledge base |
| RETRIEVAL_002 | RetrievalFailed | 500 | Failed to retrieve relevant documents |
| RETRIEVAL_003 | InsufficientContext | 200 | Insufficient information found to answer query |

### Validation Errors (4xx)
| Code | Error | HTTP Status | Description |
|------|-------|-------------|-------------|
| VALIDATION_000 | ValidationError | 422 | Request validation failed |
| VALIDATION_001 | InvalidQuery | 400 | Invalid query format or content |
| VALIDATION_002 | InvalidFileMetadata | 400 | Invalid file metadata provided |
| VALIDATION_003 | InvalidRequest | 400 | Invalid request parameters |

---

## Authentication

**Current Implementation:** No authentication (single-user mode)

For production deployment, would require:
- JWT tokens or API keys
- Header format: `Authorization: Bearer <token>`

---

## Interactive Documentation

For complete API documentation with:
- Detailed request/response schemas
- Field-level validation rules
- Interactive endpoint testing
- Code generation for multiple languages
- Real-time error validation

**Visit:** http://localhost:8000/docs (when application is running)

**Alternative UI:** http://localhost:8000/redoc (ReDoc interface)

**Note:** Documentation endpoints only available when `ENABLE_DEBUG_ENDPOINTS=true`

---

## Need Help?

- [Installation Guide](INSTALLATION.md) - Setup instructions
- [Architecture Guide](ARCHITECTURE.md) - System design
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [Docker Reference](DOCKER.md) - Container management

---

[← Back to Main README](../README.md)