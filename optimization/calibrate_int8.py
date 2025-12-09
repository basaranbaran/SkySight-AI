import tensorrt as trt
import os
import cv2
import glob
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=8, height=640, width=640):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.training_data = glob.glob(os.path.join(training_data, '*.jpg')) + \
                             glob.glob(os.path.join(training_data, '*.png'))
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.current_index = 0
        
        # Allocate device memory for batch
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * self.height * self.width * 4) # 4 bytes per float32

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.training_data):
            return None

        batch = []
        for i in range(self.batch_size):
            img_path = self.training_data[self.current_index + i]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.width, self.height))
            # Preprocessing: Convert to RGB, Normalize 0-1, Transpose (HWC->CHW)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            batch.append(img)
        
        self.current_index += self.batch_size
        
        # Make contiguous and copy to device
        batch_np = np.ascontiguousarray(np.array(batch).ravel())
        cuda.memcpy_htod(self.device_input, batch_np)
        
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it to skip calibration.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

if __name__ == "__main__":
    print("This module defines the YOLOEntropyCalibrator class.")
    print("Usage: Imported by build_trt_engine.py during INT8 engine building.")
