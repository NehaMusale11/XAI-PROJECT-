# Production Dockerfile for XAI Post-hoc Explanation Dashboard
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and pre-trained model artifacts
COPY . .

# Expose default port
EXPOSE 5000

# Health check container probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Production WSGI runner
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:${PORT}"]
