# ================================
# AI Risk Agent - Dockerfile (Stable Build)
# ================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

# Install necessary system dependencies for pandas/numpy/matplotlib
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y \
        libfreetype6-dev \
        libpng-dev \
        libopenblas-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]