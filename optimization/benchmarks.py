import time
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import get_detector

def run_benchmark(model_path, backend, video_path, num_frames=100, warmup=10, img_size=640):
    print(f"Benchmarking {backend} with {model_path} (sz={img_size})...")
    
    detector = get_detector(backend, model_path, img_size)
    cap = cv2.VideoCapture(video_path)
    
    latencies = []
    gpu_mems = []
    
    frame_count = 0
    while cap.isOpened() and frame_count < num_frames + warmup:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Warmup
        if frame_count < warmup:
            detector(frame)
            frame_count += 1
            print(f"Warmup {frame_count}/{warmup}", end='\r')
            continue
            
        # Specialize timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        detector(frame)
        end_event.record()
        torch.cuda.synchronize()
        
        latency = start_event.elapsed_time(end_event) # milliseconds
        latencies.append(latency)
        
        # Memory check
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**2 # MB
            gpu_mems.append(mem)
            
        frame_count += 1
        print(f"Frame {frame_count-warmup}/{num_frames} - Latency: {latency:.2f}ms", end='\r')
        
    cap.release()
    print("\nDone.")
    
    return {
        "backend": backend,
        "avg_latency": np.mean(latencies),
        "p50_latency": np.percentile(latencies, 50),
        "p95_latency": np.percentile(latencies, 95),
        "p99_latency": np.percentile(latencies, 99),
        "fps": 1000.0 / np.mean(latencies),
        "avg_gpu_mem": np.mean(gpu_mems) if gpu_mems else 0,
        "latencies": latencies
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--pytorch_model', type=str, default=None, help='Path to PyTorch model (e.g. models/visdrone_v8s.pt)')
    parser.add_argument('--onnx_model', type=str, default=None, help='Path to ONNX model')
    parser.add_argument('--trt_engine', type=str, default=None, help='Path to TensorRT engine')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference size')
    args = parser.parse_args()
    
    results = []
    
    # Check files
    if args.pytorch_model and os.path.exists(args.pytorch_model):
        results.append(run_benchmark(args.pytorch_model, 'pytorch', args.video, img_size=args.imgsz))
        
    if args.onnx_model and os.path.exists(args.onnx_model):
        results.append(run_benchmark(args.onnx_model, 'onnx', args.video, img_size=args.imgsz))
        
    if args.trt_engine and os.path.exists(args.trt_engine):
        results.append(run_benchmark(args.trt_engine, 'tensorrt', args.video, img_size=args.imgsz))
        
    # Save Report
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nRESULTS SUMMARY:")
    print(f"{'Backend':<15} | {'FPS':<10} | {'Latency (ms)':<15} | {'Speedup':<10}")
    print("-" * 60)
    
    base_fps = results[0]['fps'] if results else 1.0
    for res in results:
        speedup = res['fps'] / base_fps
        print(f"{res['backend']:<15} | {res['fps']:.2f}      | {res['avg_latency']:.2f}          | {speedup:.2f}x")
        
    # Plotting
    backends = [r['backend'] for r in results]
    fps_vals = [r['fps'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(backends, fps_vals, color=['blue', 'green', 'red'])
    plt.title('Inference Speed Comparison')
    plt.ylabel('FPS')
    plt.savefig('benchmark_fps_comparison.png')
    print("Graph saved to benchmark_fps_comparison.png")

if __name__ == '__main__':
    main()
