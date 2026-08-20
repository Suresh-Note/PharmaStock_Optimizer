FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports (Streamlit: 8501, FastAPI: 8000)
EXPOSE 8501 8000

# Default command (overridden by docker-compose per service)
CMD ["streamlit", "run", "app.py", \
     "--server.headless", "true", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0"]
