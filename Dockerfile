FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY static ./static
# Cloud Run injects $PORT. The app fits and caches its model at start-up (~3 s).
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
