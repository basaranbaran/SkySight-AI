import time
import json
import logging
import numpy as np
from collections import deque

try:
    import pynvml
    HAS_NVML = True
    pynvml.nvmlInit()
except:
    HAS_NVML = False

class PerformanceMonitor:
    def __init__(self, window_size=100, log_file="monitoring.log"):
        self.window_size = window_size
        
        self.latency_history = deque(maxlen=window_size)
        self.fps_history = deque(maxlen=window_size)
        self.gpu_util_history = deque(maxlen=window_size)
        
        self.frame_count_sec = 0
        self.last_fps_time = time.time()
        
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(message)s')
        
    def log_metric(self, batch_size, inference_time_sec, num_detections):
        latency_ms = inference_time_sec * 1000.0
        self.latency_history.append(latency_ms)
        
        self.frame_count_sec += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        
        if elapsed >= 1.0:
            fps = self.frame_count_sec / elapsed
            self.fps_history.append(fps)
            self.frame_count_sec = 0
            self.last_fps_time = now
            
            if HAS_NVML:
                self.gpu_util_history.append(self.get_gpu_utilization())

    def update(self):
        # Placeholder for periodic tasks if needed
        pass

    def get_gpu_utilization(self):
        if not HAS_NVML: return 0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except: return 0
            
    def get_memory_usage(self):
        if not HAS_NVML: return 0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem.used / 1024**2 # MB
        except: return 0

    def get_summary(self):
        if not self.latency_history:
            return {
                "status": "waiting_for_data",
                "avg_fps": 0,
                "gpu_util": 0
            }
            
        current_lat = self.latency_history[-1]
        avg_lat = np.mean(self.latency_history)
        p50 = np.percentile(self.latency_history, 50)
        p95 = np.percentile(self.latency_history, 95)
        p99 = np.percentile(self.latency_history, 99)
        
        current_fps = self.fps_history[-1] if self.fps_history else 0
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        current_gpu = self.gpu_util_history[-1] if self.gpu_util_history else 0
        
        return {
            "timestamp": time.time(),
            "avg_fps": round(avg_fps, 1),
            "current_fps": round(current_fps, 1),
            "latency_last": round(current_lat, 2),
            "latency_avg": round(avg_lat, 2),
            "latency_p50": round(p50, 2),
            "latency_p95": round(p95, 2),
            "latency_p99": round(p99, 2),
            "gpu_util": int(current_gpu),
            "gpu_mem_mb": int(self.get_memory_usage())
        }
