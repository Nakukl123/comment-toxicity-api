# Small official Python base image
FROM python:3.11-slim

# Don't write .pyc files; flush logs straight to the console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Run as a non-root user (security best practice; Hugging Face Spaces expects uid 1000)
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

# Install dependencies first so Docker caches this layer between code changes
COPY --chown=appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser app ./app
COPY --chown=appuser model_store ./model_store

USER appuser
EXPOSE 7860

# Docker restarts/flags the container if the API stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
