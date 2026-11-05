# Visenlo — optional local Docker container
# Still runs entirely on the user's machine (no external service).
# Build & run: docker compose up --build  -> http://localhost:7860
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0

# System deps for opencv / onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-base.txt ./
# CPU onnxruntime in the container by default (GPU: nvidia runtime + requirements-gpu.txt)
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py app.css ./
COPY visenlo ./visenlo
COPY scripts ./scripts

# Models are installed explicitly and mounted as a local volume.
RUN mkdir -p models outputs

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
