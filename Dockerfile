FROM python:3.13-slim
 
WORKDIR /app
 
# Install uv
RUN pip install uv --no-cache-dir
 
# Copy dependency files first (layer cache)
COPY pyproject.toml .
RUN uv pip install --system --no-cache -e .
 
# Copy data files
COPY data/market.duckdb ./data/market.duckdb
 
# Copy application code
COPY app/      ./app/
COPY static/   ./static/
COPY data/     ./data/
 
# Create artifacts directory
RUN mkdir -p artifacts
 
# Cloud Run listens on PORT env var
ENV PORT=8080
EXPOSE 8080
 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]