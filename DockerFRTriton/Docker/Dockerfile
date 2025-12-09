FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Workdir inside container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy code and model repository
COPY . /app

# Install Python deps for FastAPI wrapper
RUN pip install --no-cache-dir -r requirements.txt

# Default ports for Triton (8000/8001/8002) and FastAPI (3000)
EXPOSE 8000 8001 8002 3000

# Launch Triton and FastAPI together
CMD ["bash", "/app/Docker/start.sh"]
