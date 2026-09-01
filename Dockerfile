# ReactorX — optional local Docker container
# Still runs entirely on the user's machine (no external service).
# Build & run: docker compose up --build  -> http://localhost:7860
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0

# System deps for opencv / onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# Use CPU onnxruntime in container by default (GPU: use nvidia runtime + onnxruntime-gpu)
RUN pip install --upgrade pip && \
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true && \
    pip install -r requirements.txt && \
    pip uninstall -y onnxruntime 2>/dev/null || true && \
    pip install onnxruntime

COPY app.py ./ 
COPY reactorx ./reactorx
COPY scripts ./scripts

# Models are mounted as volume; buffalo_l auto-downloads on first run
RUN mkdir -p models outputs

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
