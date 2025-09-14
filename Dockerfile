# Base image with CUDA runtime and Python
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Prevent interactive tzdata
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirement files first for caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Create directory for static models to persist inside container
RUN mkdir -p /app/static/models

# Expose port
EXPOSE 8000

# Environment for Django
ENV PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=videoup.settings \
    PORT=8000 \
    WEB_CONCURRENCY=1 \
    REQUIRE_CUDA=1

# Healthcheck to verify server responds (expects compose to map port)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/api/info/ || exit 1

# Entrypoint script to run migrations and start server
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
