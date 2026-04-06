FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache-friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY bot.py censor.py db.py settings.py ./

# Pre-warm the NudeNet ONNX model so the first runtime request is fast.
# This downloads the weights into the image layer at build time.
RUN python -c "from nudenet import NudeDetector; NudeDetector()"

# Health-check uses the built-in HTTP endpoint (configurable via HEALTH_PORT).
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "bot.py"]
