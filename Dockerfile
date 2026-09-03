# Dockerfile — builds just the API service (models + src/api), not the notebooks.
FROM python:3.12-slim

WORKDIR /app

# System deps some ML wheels need at runtime (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only copy what the API actually needs
COPY models/ ./models/
COPY src/api/ ./src/api/

WORKDIR /app/src/api

# Render (and most PaaS) inject $PORT — default to 8000 for local `docker run`
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
