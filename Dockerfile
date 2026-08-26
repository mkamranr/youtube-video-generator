FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching), then pre-fetch the pinned
# ffmpeg/ffprobe binaries so the container needs no download at runtime.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "from static_ffmpeg import run; run.get_or_fetch_platform_executables_else_raise()"

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/options', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
