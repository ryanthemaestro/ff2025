# Use slim Python base image
FROM python:3.10-slim

# Avoid interactive prompts during apt installs
ARG DEBIAN_FRONTEND=noninteractive

# Environment settings for Python and headless operation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen \
    DISPLAY=""

# Workdir inside the container
WORKDIR /app

# Install system deps required by some Python packages (e.g., CatBoost needs libgomp)
# Then install Python dependencies
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc g++ libgomp1 \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the source code
COPY . .

# Expose a default port (Render will provide $PORT at runtime)
EXPOSE 10000

# Start the app via Gunicorn, binding to the Render-provided $PORT
CMD ["sh", "-c", "exec gunicorn wsgi:app --workers 2 --threads 2 --timeout 180 --bind 0.0.0.0:${PORT:-10000}"]
