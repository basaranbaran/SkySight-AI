import cv2
import requests
import argparse
import time
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description="VisDrone API Demo Client")
    parser.add_argument('--video', type=str, default='sample.mp4', help='Path to video')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000/detect', help='API Endpoint')
    parser.add_argument('--model', type=str, default=None, help='Specific model to use (trt_fp16, trt_int8, onnx, pytorch)')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    # Sync FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120: video_fps = 30
    frame_dur = 1.0 / video_fps
    
    font = cv2.FONT_HERSHEY_SIMPLEX # Define font globally for main
    is_fullscreen = False
    fps_avg = 0 # Initialize FPS accumulator
    
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        try:
            params = {}
            if args.model:
                params['model'] = args.model
                
            response = requests.post(
                args.api_url, 
                params=params,
                files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            )
            
            if response.status_code == 200:
                data = response.json()
                detections = data.get("detections", [])
                infer_ms = data.get("inference_time_ms", 0)
                
                for det in detections:
                    bbox = det['bbox']
                    conf = det['confidence']
                    cls = det['class_id']
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 255, 0) # Green for all
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"Class {cls} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), font, 0.5, color, 2)
                
                # Calc FPS
                t_end = time.time()
                elapsed = t_end - t_start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_avg = 0.9 * fps_avg + 0.1 * fps
                
                # Sync logic
                if elapsed > frame_dur:
                    frames_to_skip = int(elapsed / frame_dur)
                    if frames_to_skip > 0:
                        # Skip reading frames to catch up
                        for _ in range(frames_to_skip):
                            cap.read()
                
                cv2.rectangle(frame, (0, 0), (450, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"API FPS: {fps_avg:.1f} | Latency: {infer_ms:.1f}ms", 
                           (10, 25), font, 0.7, (0, 255, 255), 2)
            
            else:
                print(f"API Error: {response.status_code}")
                # Show raw frame if API fails
                
        except requests.exceptions.ConnectionError:
            print("Connection failed! Is the API server running?")
            break
            
        cv2.namedWindow("API Real-Time Client", cv2.WINDOW_NORMAL)
        if not is_fullscreen:
            cv2.resizeWindow("API Real-Time Client", 1280, 720)
            
        cv2.imshow("API Real-Time Client", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
             # Fullscreen toggle
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty("API Real-Time Client", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("API Real-Time Client", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow("API Real-Time Client", 1280, 720)
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
