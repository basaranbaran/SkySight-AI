import argparse
import sys
import os
from ultralytics import YOLO

def export_onnx(weights_path, opset=12, dynamic=True):
    """
    Exports a YOLOv8 model to ONNX format with dynamic axes.
    """
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        # Create a dummy model for testing purposes if real weights don't exist
        print("Creating a dummy YOLOv8n model for export testing...")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(weights_path)

    print(f"Exporting {weights_path} to ONNX (opset={opset}, dynamic={dynamic})...")
    
    # YOLOv8 export with dynamic axes
    # Ultralytics handles the complexity of dynamic shapes (batch, height, width) 
    # when dynamic=True.
    success = model.export(
        format='onnx',
        opset=opset,
        dynamic=dynamic,
        simplify=True  # Simplify ONNX model
    )
    
    if success:
        print(f"Export successful: {success}")
        
        # Validation
        import onnxruntime
        import numpy as np
        
        print("Validating ONNX model...")
        ort_session = onnxruntime.InferenceSession(success, providers=['CPUExecutionProvider'])
        
        # Create dummy input
        img_size = 640
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        # PyTorch Inference
        model.eval()
        with torch.no_grad():
            torch_out = model(dummy_input)[0].numpy()
            
        # ONNX Inference
        onnx_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_out = ort_session.run(None, onnx_input)[0]
        
        # Compare
        try:
            np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-4)
            print("✅ ONNX Validation Passed! Outputs match PyTorch.")
        except AssertionError as e:
            print("❌ ONNX Validation Failed! Outputs do not match.")
            print(e)
            
    else:
        print("Export failed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/latest.pt', help='Path to input weights')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    args = parser.parse_args()
    
    export_onnx(args.weights, args.opset)
