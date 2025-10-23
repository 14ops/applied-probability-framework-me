FROM python:3.9-slim

LABEL maintainer="14ops"
LABEL description="Applied Probability Framework - Professional Monte Carlo simulation system"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY src/python/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app/src/python

# Default command: run the quick demo
CMD ["python", "examples/quick_demo.py"]

