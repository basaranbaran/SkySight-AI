import time
import json
import os
import numpy as np
import pynvml
from monitoring.fps_meter import FPSMeter

class PerformanceMonitor:
    def __init__(self, log_path="monitoring/metrics.json"):
        self.fps_meter = FPSMeter()
        self.log_path = log_path
        self.metrics_buffer = []
        self.latencies = [] # Store raw latencies for histogram
        
        # Ensure dir exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Init NVML for GPU stats
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assume single GPU
            self.gpu_available = True
        except Exception as e:
            print(f"NVML Init Failed: {e}")
            self.gpu_available = False
        
    def update(self):
        self.fps_meter.update()
        
    def log_metric(self, frame_id, processing_time, num_detections):
        # Create metric object
        current_fps = self.fps_meter.get_last_fps()
        avg_fps = self.fps_meter.get_avg_fps()
        
        self.latencies.append(processing_time * 1000) # Store in ms
        # Keep last 1000 latencies for histogram to save memory
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
        
        metric = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "fps": round(current_fps, 2),
            "avg_fps": round(avg_fps, 2),
            "processing_time": round(processing_time * 1000, 2), # ms
            "detections": num_detections
        }
        
        # Add GPU Stats if available
        if self.gpu_available:
            gpu_stats = self.get_gpu_stats()
            metric.update(gpu_stats)
        
        self.metrics_buffer.append(metric)
        
        # Flush every 100 frames to avoid disk I/O lag
        if len(self.metrics_buffer) >= 100:
            self.save_metrics()
            
    def get_gpu_stats(self):
        if not self.gpu_available:
            return {}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return {
                "gpu_util": util.gpu,
                "gpu_mem_used": mem.used / 1024**2, # MB
                "gpu_mem_total": mem.total / 1024**2
            }
        except:
            return {}
            
    def save_metrics(self):
        with open(self.log_path, 'a') as f:
            for m in self.metrics_buffer:
                f.write(json.dumps(m) + "\n")
        self.metrics_buffer = []

    def get_summary(self):
        # Calculate Latency Percentiles
        if self.latencies:
            p50 = np.percentile(self.latencies, 50)
            p90 = np.percentile(self.latencies, 90)
            p95 = np.percentile(self.latencies, 95)
            p99 = np.percentile(self.latencies, 99)
        else:
            p50 = p90 = p95 = p99 = 0
            
        summary = {
            "avg_fps": self.fps_meter.get_avg_fps(),
            "latency_p50": round(p50, 2),
            "latency_p90": round(p90, 2),
            "latency_p95": round(p95, 2),
            "latency_p99": round(p99, 2)
        }
        
        if self.gpu_available:
            summary.update(self.get_gpu_stats())
            
        return summary
