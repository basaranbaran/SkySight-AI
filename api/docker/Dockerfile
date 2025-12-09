# Use NVIDIA TensorRT runtime as base
FROM nvcr.io/nvidia/tensorrt:23.08-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app

# Expose API port
EXPOSE 8000

# Start server using module method
CMD ["python3", "-m", "api.server"]
