FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv sync

# Expose port
EXPOSE 8000

# Run seeding and then the application
CMD ["sh", "-c", "uv run python download_models.py && uv run python seed_catalog.py && uv run python seed_economic_cache.py && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"]
