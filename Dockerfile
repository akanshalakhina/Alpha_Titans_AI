# Alpha Titans — Content Engine
#
#   docker build -t alpha-titans .
#   docker run -p 8000:8000 -v "$PWD/outputs:/app/outputs" -v "$PWD/uploads:/app/uploads" alpha-titans
#
# ffmpeg is installed here rather than assumed, because the pipeline is
# entirely ffmpeg/ffprobe work and the image is useless without it.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AT_HOST=0.0.0.0 \
    AT_PORT=8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first so they cache across source changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/
COPY server/   ./server/
COPY frontend/ ./frontend/
COPY .env.example ./

# renders and uploads live here; mount volumes over them to keep them
RUN mkdir -p uploads outputs data/music

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "/app/server", \
     "--host", "0.0.0.0", "--port", "8000"]
