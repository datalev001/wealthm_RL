FROM python:3.11-slim

WORKDIR /app

# copy source
COPY . /app

# deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.2 \
        fastapi==0.111.0 \
        uvicorn==0.29.0 \
        grpcio==1.63.0 \
        prometheus-client==0.20.0 \
        pandas==2.2.2 \
        numpy==1.26.4

# expose service
CMD ["uvicorn", "inference.serve:app", "--host", "0.0.0.0", "--port", "8000"]