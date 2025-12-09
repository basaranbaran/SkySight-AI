import torch
import numpy as np
import cv2
import time
import os
import onnxruntime as ort
from abc import ABC, abstractmethod
from inference.utils import letterbox, non_max_suppression, scale_boxes

class BaseDetector(ABC):
    def __init__(self, model_path, device='cuda', conf_thres=0.25, iou_thres=0.45, img_size=640):
        self.model_path = model_path
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.stride = 32
        self.img_size = img_size

    @abstractmethod
    def infer(self, img):
        pass

    def preprocess(self, img0):
        # Resize and pad
        # auto=False required for static TensorRT shapes
        img, self.ratio, self.dwdh = letterbox(img0, new_shape=self.img_size, stride=self.stride, auto=False)
        
        # Standard preprocessing: Transpose, Contiguous, Float, Scale
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)
            
        return img, img0

    def postprocess(self, preds, img, img0):
        # NMS
        preds = non_max_suppression(preds, self.conf_thres, self.iou_thres)
        
        results = []
        for i, det in enumerate(preds):
            if len(det):
                # Rescale to original resolution
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                results.append(det)
            else:
                results.append(torch.zeros((0, 6)))
        return results[0] # Return detections for the single image

    def __call__(self, img):
        t0 = time.time()
        
        # Batch support logic
        is_batch = isinstance(img, list)
        
        if is_batch:
            # Process list of images
            det_batch = []
            t_infers = 0
            for i, im in enumerate(img):
                img_pre, img_src = self.preprocess(im)
                # Infer single
                preds = self.infer(img_pre)
                # Postprocess
                dets = self.postprocess(preds, img_pre, img_src)
                det_batch.append(dets)
            t_infer = time.time() - t0
            return det_batch, t_infer
        else:
            # Single image path
            img_pre, img_src = self.preprocess(img)
            preds = self.infer(img_pre)
            t_infer = time.time() - t0
            
            dets = self.postprocess(preds, img_pre, img_src)
            
            return dets, t_infer

class PyTorchDetector(BaseDetector):
    def __init__(self, model_path, device='cuda', conf_thres=0.25):
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.")
            device = 'cpu'
        super().__init__(model_path, device, conf_thres)
        self.backend = 'pytorch'
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model_internal = self.model.model.to(device)
        self.model_internal.eval()
        
    def infer(self, img):
        with torch.no_grad():
            return self.model_internal(img)[0]

class ONNXDetector(BaseDetector):
    def __init__(self, model_path, device='cuda', conf_thres=0.25, img_size=640):
        super().__init__(model_path, device, conf_thres, img_size=img_size)
        self.backend = 'onnx'
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
    def infer(self, img):
        img_np = img.cpu().numpy()
        pred = self.session.run(None, {self.input_name: img_np})[0]
        # Standardize output to torch
        return torch.from_numpy(pred).to(self.device)

class TensorRTDetector(BaseDetector):
    def __init__(self, model_path, device='cuda', conf_thres=0.25, img_size=640):
        super().__init__(model_path, device, conf_thres, img_size=img_size)
        self.backend = 'tensorrt'
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.INFO)
        with open(model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}. Version mismatch likely.")

        self.context = self.engine.create_execution_context()
        
        self.input_name = None
        self.output_name = None
        
        # Identify I/O tensors
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
                
    def infer(self, img):
        # Prepare GPU buffers
        self.context.set_input_shape(self.input_name, tuple(img.shape))
        
        # Handle dynamic shapes from engine
        out_shape = list(self.engine.get_tensor_shape(self.output_name))
        batch_size = img.shape[0]
        
        # Resolve dynamic dimensions
        if out_shape[0] == -1: out_shape[0] = batch_size
        if out_shape[2] == -1: out_shape[2] = 33600 # Fallback for VisDrone 1280
             
        output = torch.zeros(tuple(out_shape), device=self.device, dtype=torch.float32)
        
        # Bind and Execute
        self.context.set_tensor_address(self.input_name, int(img.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
        
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        
        return output

def get_detector(backend='pytorch', model_path='models/latest.pt', img_size=640):
    if backend == 'pytorch':
        return PyTorchDetector(model_path, conf_thres=0.25) # base ignored
    elif backend == 'onnx':
        return ONNXDetector(model_path, conf_thres=0.25, img_size=img_size)
    elif backend == 'tensorrt':
        return TensorRTDetector(model_path, conf_thres=0.25, img_size=img_size)
    else:
        raise ValueError(f"Unknown backend: {backend}")
