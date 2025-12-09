import cv2
import argparse
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.video_engine import VideoEngine

def main():
    parser = argparse.ArgumentParser(description="VisDrone Demo Utils")
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='Path to video file')
    parser.add_argument('--model', type=str, default='models/latest.pt', help='Path to model')
    parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'onnx', 'tensorrt'])
    parser.add_argument('--imgsz', type=int, default=1280, help='Inference size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--title', type=str, default='VISDRONE AI ANALYTICS', help='Custom title for the HUD')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        print("Please place a video file (e.g., sample.mp4) in this folder or specify path with --video")
        return

    print(f"Starting Demo...")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Resolution: {args.imgsz}x{args.imgsz}")
    print("-" * 40)
    print("Press 'q' to quit.")
    
    # Initialize Engine
    engine = VideoEngine(
        source=args.video,
        model_path=args.model,
        backend=args.backend,
        img_size=args.imgsz
    )
    
    # Sync playback to video FPS
    cap_temp = cv2.VideoCapture(args.video)
    video_fps = cap_temp.get(cv2.CAP_PROP_FPS)
    cap_temp.release()
    
    if video_fps <= 0 or video_fps > 120: video_fps = 30
    target_delay = 1.0 / video_fps
    print(f"Original Video FPS: {video_fps:.2f} (Target Delay: {target_delay*1000:.1f}ms)")
    
    engine.start()
    fps_avg = 0
    frame_cnt = 0
    is_fullscreen = False
    
    cv2.namedWindow("VisDrone AI System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VisDrone AI System", 1280, 720)
    
    try:
        for frame in engine.get_results():
            t_start_loop = time.time()
            frame_cnt += 1
            
            # Draw HUD
            h, w = frame.shape[:2]
            
            # Top Bar
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, args.title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Stats
            if engine.fps_stats:
                curr_fps = engine.fps_stats[-1]
                fps_avg = (fps_avg * 0.9) + (curr_fps * 0.1)
                
            cv2.putText(frame, f"FPS: {int(fps_avg)}", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Backend: {args.backend.upper()}", (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Show
            # Aspect ratio preservation
            cv2.imshow("VisDrone AI System", frame)
            
            t_process_loop = time.time() - t_start_loop
            sleep_time = target_delay - t_process_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                # Fullscreen toggle
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty("VisDrone AI System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty("VisDrone AI System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("VisDrone AI System", 1280, 720)
                
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        cv2.destroyAllWindows()
        print("\nDemo Finished.")

if __name__ == '__main__':
    main()
