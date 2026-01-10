FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY feature_store/ feature_store/

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
