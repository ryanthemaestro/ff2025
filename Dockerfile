# Use official Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libatlas-base-dev \
        gfortran \
        gcc \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Set fallback PORT
ENV PORT 8000

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app/

# Default command with shell expansion for PORT (defaults to 8000)
CMD ["sh", "-c", "gunicorn functions.api:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 300 --log-level debug"] 