import cv2
import time
import threading
import queue
from inference.detector import get_detector
from inference.tracker import get_tracker
from inference.fusion import FusionEngine

class VideoEngine:
    def __init__(self, source, model_path, backend='pytorch', img_size=640):
        if isinstance(source, int):
            # Use DirectShow for compatibility
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
            
        # Store params for lazy initialization in thread
        self.backend = backend
        self.model_path = model_path
        self.img_size = img_size
        self.detector = None 
        
        self.tracker = get_tracker('bytetrack')
        self.fusion = FusionEngine()
        self.enable_tracker = True # Default on
        
        self.frame_queue = queue.Queue(maxsize=5) # Increased buffer
        self.result_queue = queue.Queue(maxsize=5)
        self.stopped = False
        
        # Config
        self.detect_interval = 1 
        self.frame_count = 0
        self.fps_stats = []
        
        # Determine if source is file or stream
        self.is_file = not isinstance(source, int)
    
    def start(self):
        self.t_capture = threading.Thread(target=self.capture_loop)
        self.t_process = threading.Thread(target=self.process_loop)
        
        # Daemonize threads so they die with main
        self.t_capture.daemon = True
        self.t_process.daemon = True
        
        self.t_capture.start()
        self.t_process.start()
        
    def capture_loop(self):
        while not self.stopped and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
                
            # File: blocking I/O, Webcam: non-blocking
            if self.is_file:
                self.frame_queue.put(frame)
            else:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # If full, drop oldest to avoid lag (webcam mode)
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass
            
            # Minimal sleep to yield GIL
            time.sleep(0.0001)

    def process_loop(self):
        try:
            # Init detector in thread for CUDA context
            if self.detector is None:
                 print(f"Loading detector ({self.backend}) in process thread...")
                 self.detector = get_detector(self.backend, self.model_path, self.img_size)

            tracks = []
            prev_time = time.time()
            
            while not self.stopped:
                if self.frame_queue.empty():
                    time.sleep(0.001)
                    continue
                    
                frame = self.frame_queue.get()
                self.frame_count += 1
                
                # Logic: Detect every N frames, Track always
                if self.frame_count % self.detect_interval == 0:
                    dets, t_infer = self.detector(frame)
                    
                    if self.enable_tracker:
                        # Drift Check
                        current_tracks = [t.tlwh for t in self.tracker.tracked_stracks if t.state == 2]
                        current_tracks_fmt = [[t[0], t[1], t[2], t[3]] for t in current_tracks]
                        
                        if hasattr(dets, 'cpu'): dets_cpu = dets.cpu().numpy()
                        else: dets_cpu = dets
                        det_boxes = dets_cpu[:, :4] if len(dets_cpu) > 0 else []
                        
                        if len(current_tracks_fmt) > 0 and len(det_boxes) > 0:
                            is_drift = self.fusion.check_drift(det_boxes, current_tracks_fmt)
                            if is_drift:
                                print(f"Frame {self.frame_count}: Drift Detected! Reinitializing Tracker.")
                                self.tracker = get_tracker('bytetrack')
                        
                        tracks = self.tracker.update(dets, frame)
                    else:
                         tracks = []
                else:
                    # Only update tracker without new detections if supported (future improvement)
                    pass 
                
                # Calculate FPS
                curr_time = time.time()
                if (curr_time - prev_time) > 0:
                    fps = 1.0 / (curr_time - prev_time)
                else:
                    fps = 0
                prev_time = curr_time
                
                self.fps_stats.append(fps)
                    
                # Visualization
                res_frame = self.draw_tracks(frame, tracks)
                
                cv2.putText(res_frame, f"FPS: {fps:.1f} Dets: {len(dets) if 'dets' in locals() else 0} Tracks: {len(tracks)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if not self.result_queue.full():
                    self.result_queue.put(res_frame)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in process_loop: {e}")
            self.stopped = True
                
    def draw_tracks(self, frame, tracks):
        for t in tracks:
            # tlwh = t[:4]
            x, y, w, h = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            tid = int(t[4])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tid}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def get_results(self):
        while not self.stopped:
            if not self.result_queue.empty():
                yield self.result_queue.get()
            else:
                time.sleep(0.001)

    def stop(self):
        self.stopped = True
        self.t_capture.join()
        self.t_process.join()
        self.cap.release()
        
        if self.fps_stats:
            avg_fps = sum(self.fps_stats) / len(self.fps_stats)
            print("-" * 30)
            print(f"PERFORMANCE REPORT:")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total Frames Processed: {len(self.fps_stats)}")
            print("-" * 30)

if __name__ == "__main__":
    pass
