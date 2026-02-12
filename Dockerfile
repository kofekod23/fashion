# CPU-only Dockerfile (lighter, no CUDA dependencies)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEVICE=cpu

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install Python dependencies (CPU-only PyTorch)
RUN pip install --no-cache-dir -e .

# Create data directory for datasets
RUN mkdir -p /app/data

# Expose port for web server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the web server with multiple workers
CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
