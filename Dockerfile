# Image de déploiement de la webapp trainedml.
#
#   docker build -t trainedml-webapp .
#   docker run -p 8000:8000 trainedml-webapp

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py ./
COPY static ./static

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
