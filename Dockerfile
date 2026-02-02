# NautilusTrader + RL Agents Docker Image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Install IB API - use version compatible with NautilusTrader 1.221
RUN pip install nautilus_ibapi==10.30.1

# Copy application code
COPY config/ ./config/
COPY strategies/ ./strategies/
COPY gym_env/ ./gym_env/
COPY training/ ./training/
COPY validation/ ./validation/
COPY live/ ./live/
COPY monitoring/ ./monitoring/
COPY data/adapters/ ./data/adapters/
COPY main.py .
COPY ib_test.py .

# Create necessary directories
RUN mkdir -p /app/data/catalog /app/models /app/logs

# Create non-root user for security
RUN useradd -m -u 1000 nautilus && \
    chown -R nautilus:nautilus /app
USER nautilus

# Health check endpoint port
EXPOSE 8000

# Default command
CMD ["python", "main.py"]
