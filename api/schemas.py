from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int
    class_name: str = ""

class DetectionResponse(BaseModel):
    detections: List[BoundingBox]
    inference_time_ms: float
    fps: float
    device: str

class MetricsResponse(BaseModel):
    latency_avg_ms: float
    latency_p95_ms: float
    throughput_fps: float
    gpu_utilization: float
    memory_usage_mb: float
