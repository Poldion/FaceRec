# Gesichtserkennung fuer die Tuerklingel. Gebaut ueber facerec/docker-compose.yml.
FROM python:3.12-slim
LABEL org.opencontainers.image.title="FaceRec" \
      org.opencontainers.image.source="https://github.com/Poldion/FaceRec" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# InsightFace "buffalo_l" (offizielles Release), davon nur Gesichtsfinder + Gesichtsabdruck
ARG MODELS_URL=https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
ARG MODELS_SHA256=80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f
RUN python -c "import hashlib, io, sys, urllib.request, zipfile; \
d = urllib.request.urlopen('$MODELS_URL', timeout=600).read(); \
h = hashlib.sha256(d).hexdigest(); \
sys.exit('Pruefsumme falsch: ' + h) if h != '$MODELS_SHA256' else None; \
z = zipfile.ZipFile(io.BytesIO(d)); [z.extract(n, '/models') for n in ('det_10g.onnx', 'w600k_r50.onnx')]"

COPY engine.py tracker.py app.py i18n.py ./
COPY locales ./locales
COPY static ./static

ENV FACEREC_MODELS=/models \
    FACEREC_DATA=/data \
    FACEREC_PORT=8191 \
    PYTHONUNBUFFERED=1
EXPOSE 8191

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8191/health', timeout=4)"]

CMD ["python", "app.py"]
