FROM python:3.11-slim

LABEL maintainer="Yevgeniy Yermoshin, yev.developer@gmail.com"
LABEL version="1.0"
LABEL description="Dockerfile for FastAPI application with ML model serving"

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
    

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

WORKDIR /app/

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


CMD ["uvicorn", "--host=0.0.0.0", "src.main:app", "--port=5000"]

