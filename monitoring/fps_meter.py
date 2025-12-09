import time
import collections

class FPSMeter:
    def __init__(self, buffer_len=100):
        self.buffer = collections.deque(maxlen=buffer_len)
        self.prev_time = time.time()
        
    def update(self):
        curr_time = time.time()
        delta = curr_time - self.prev_time
        if delta > 0:
            fps = 1.0 / delta
            self.buffer.append(fps)
        self.prev_time = curr_time
        
    def get_avg_fps(self):
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)
    
    def get_last_fps(self):
        if not self.buffer:
            return 0.0
        return self.buffer[-1]
