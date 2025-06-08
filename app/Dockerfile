FROM python:3.11-slim


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

