# 🦅 SkySight-AI - Advanced Aerial Object Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorRT](https://img.shields.io/badge/TensorRT-8.6-green.svg)
![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-teal.svg)

## 📌 Project Overview
This package implements a **High-Performance Object Detection & Tracking System** specifically optimized for aerial (drone) imagery. It addresses the challenge of detecting small objects (humans, vehicles) in high-resolution video streams (1280px) using **TensorRT (FP16/INT8)** optimization.

**Key Features:**
*   **High-Res Inference:** 1280x1280 input for maximum recall.
*   **TensorRT Optimized:** **132 FPS** on RTX 3060 (FP16).
*   **Real-Time Dashboard:** Web-based monitoring for FPS, Latency, and GPU Stats.
*   **Production Ready:** Dockerized microservice architecture.

## 📂 Package Structure

```bash
📦 .                         # Root (formerly 'main')
├── 📂 api/                  # FastAPI Server, Dashboard & Dockerfile
├── 📂 inference/            # Detection & Tracking Engines
├── 📂 optimization/         # TRT Conversion & Benchmarking Scripts
├── 📂 monitoring/           # Performance Logger
├── 📂 training/             # Training Scripts
├── 📂 (Extra) Demos/        # Additional Demo Scripts (Visual & API)
└── 📂 tests/                # Internal Tests
```

## 🚀 Installation & Usage

### 1. Local Setup (Windows/Linux)
**Prerequisites:** NVIDIA GPU + CUDA 11+.

```bash
# 1. Install Dependencies
pip install -r requirements.txt
pip install ultralytics onnxruntime-gpu

# 2. Run the API Server
# Note: Run this command from this directory
python -m api.server
```
*The server will start at `http://localhost:8000`. Open this URL to see the Real-Time Dashboard.*

### 2. Docker Deployment (Recommended)
Build and run the containerized service.

```bash
# 1. Build Image
docker build -f api/docker/Dockerfile -t visdrone-api .

# 2. Run Container (Mount your 'models' folder if external)
docker run --gpus all -p 8000:8000 -v ${PWD}/../models:/app/models visdrone-api
```

## 🛠️ Advanced Usage & Benchmarking

### 1. Running Demos (Local)
Visual demos are located in the `(Extra) Demos` folder.

**Visual Demo (OpenCV Window):**
```bash
python "(Extra) Demos/demo.py" --video ../sample/sample-video.mp4 --backend tensorrt
```

**API Client (Docker Test):**
```bash
python "(Extra) Demos/demo_api_client.py" --video ../sample/sample-video.mp4 --model trt_docker_fp16
```

### 2. Running Individual Benchmarks
You can test specific models/backends individually to measure raw performance.

**Test TensorRT FP16 (Best Performance):**
```bash
python optimization/benchmarks.py --video ../sample/sample-video.mp4 --trt_engine ../models/visdrone_v8s_fp16.engine --imgsz 1280
```

## 🏆 Performance Benchmarks
Benchmarked on NVIDIA RTX 3060 (1280px Input).

| Backend | FPS (Avg) | Latency (ms) | Status |
| :--- | :--- | :--- | :--- |
| **PyTorch (Baseline)** | 38.4 FPS | 26.0 ms | Baseline |
| **TensorRT (FP16)** | **131.9 FPS** | **7.5 ms** | ✅ **Preferred** |
| **Docker API (FP16)** | ~110 FPS | 9.1 ms | Production Ready |

## 📝 API Endpoints
*   `POST /detect`: Inference endpoint. Upload a frame, get bbox JSON.
*   `GET /metrics`: Live GPU & Latency stats.
*   `GET /health`: System health check.
