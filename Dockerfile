# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for document processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    # Required for complex file conversion (needed by pypandoc/unstructured)
    pandoc \
    curl \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from backend folder
COPY backend/requirements-frozen.txt .

RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (torch will be installed as dependency)
#RUN pip install -r requirements-frozen.txt

RUN cat requirements-frozen.txt \
    | grep -v -e '^torch==' -e '^torchvision==' -e '^nvidia_' \
    | pip install -r /dev/stdin

# Copy application code from backend folder
COPY backend/main.py .
COPY backend/config.py .
COPY backend/services.py .
COPY backend/utils.py .
COPY backend/exceptions.py .

# Copy test files (optional, for running tests in container)
COPY backend/tests/ ./tests/

# Create directories for data persistence
RUN mkdir -p /app/qdrant_storage /app/temp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]