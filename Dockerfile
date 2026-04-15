# Stage 1: Build the backend
FROM python:3.11-slim as backend

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# Note: In production, consider switching 'tensorflow' to 'tensorflow-cpu' in requirements.txt
# to keep the image size smaller (~500MB vs ~1.5GB).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Start the application using Gunicorn
# Bind to the dynamic $PORT provided by the environment
CMD ["sh", "-c", "gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.app:app --bind 0.0.0.0:$PORT"]
