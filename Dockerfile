FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-submission"
LABEL description="Email Triage OpenEnv — real-world email triage environment"
LABEL version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/        ./app/
COPY data/       ./data/
COPY openenv.yaml .
COPY inference.py .

# Create non-root user for security
RUN useradd -m -u 1000 envuser && chown -R envuser:envuser /app
USER envuser

# Environment defaults (override at runtime)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV ENV_BASE_URL="http://localhost:7860"
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 7860

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]