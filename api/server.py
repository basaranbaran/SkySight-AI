import sys
import os
import cv2
import numpy as np
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import uvicorn

# Allow importing from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import get_detector
from inference.tracker import get_tracker
from monitoring.logger import PerformanceMonitor

# Global Variables
detectors: Dict[str, Any] = {}
tracker = None
monitor = None
default_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detectors, tracker, monitor, default_model
    print("Loading AI Models...")
    
    # Valid model candidates with their backends
    model_candidates = {
        'trt_fp16':        {'backend': 'tensorrt', 'path': os.path.join('models', 'model_fp16.engine')},
        'trt_int8':        {'backend': 'tensorrt', 'path': os.path.join('models', 'model_int8.engine')},
        'trt_docker_fp16': {'backend': 'tensorrt', 'path': os.path.join('models', 'model_docker_fp16.engine')},
        'trt_docker_int8': {'backend': 'tensorrt', 'path': os.path.join('models', 'model_docker_int8.engine')},
        'onnx':            {'backend': 'onnx',     'path': os.path.join('models', 'model.onnx')},
        'pytorch':         {'backend': 'pytorch',  'path': os.path.join('models', 'latest.pt')},
    }
    
    for name, config in model_candidates.items():
        if os.path.exists(config['path']):
            try:
                print(f"Loading {name} ({config['backend']})...")
                # Force 1280 resolution typical for VisDrone models to prevent silent failures
                det = get_detector(config['backend'], config['path'], img_size=1280)
                detectors[name] = det
                print(f"✅ Loaded {name}")
                
                # Model precedence: INT8 > FP16 > Others
                if default_model is None: default_model = name
                if name == 'trt_fp16': default_model = name
                if name == 'trt_int8': default_model = name
                
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
    
    if not detectors:
        print("Warning: No custom models found. Loading YOLOv8n fallback.")
        try:
            detectors['fallback'] = get_detector('pytorch', 'yolov8n.pt')
            default_model = 'fallback'
        except:
            print("Critical: Could not load any model.")

    tracker = get_tracker('bytetrack')
    monitor = PerformanceMonitor()
    
    print(f"System Ready. Active Models: {list(detectors.keys())}. Default: {default_model}")
    yield
    print("Shutting down.")

app = FastAPI(title="VisDrone AI API", lifespan=lifespan)

# Static Files for Dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir): os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/dashboard.html")

@app.get("/health")
def health_check():
    if not detectors:
        raise HTTPException(status_code=503, detail="No models loaded")
    return {
        "status": "live", 
        "models": list(detectors.keys()),
        "default": default_model
    }

@app.get("/metrics")
def get_metrics():
    if monitor:
        return monitor.get_summary()
    return {"error": "Monitor not initialized"}

@app.post("/detect")
async def detect(model: str = None, file: UploadFile = File(...)):
    start_time = time.time()
    
    # Select Model
    target_model = model if model else default_model
    if target_model not in detectors:
        raise HTTPException(status_code=400, detail=f"Model '{target_model}' not found. Available: {list(detectors.keys())}")
        
    active_detector = detectors[target_model]
    
    # Read Image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Inference
    detections, t_infer = active_detector(img)
    
    # Format Results
    results = []
    if hasattr(detections, 'cpu'):
        detections = detections.cpu().numpy()
        
    for det in detections:
        results.append({
            "bbox": [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
            "confidence": float(det[4]),
            "class_id": int(det[5])
        })
        
    process_time = time.time() - start_time
    
    # Update Stats
    if monitor:
        monitor.log_metric(0, process_time, len(results))
        monitor.update() 
        
    return {
        "model": target_model,
        "count": len(results),
        "inference_time_ms": t_infer * 1000,
        "total_time_ms": process_time * 1000,
        "detections": results
    }

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
