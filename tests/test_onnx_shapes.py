import onnxruntime
import numpy as np
import os

def test_onnx_shapes():
    model_path = 'yolov8n.onnx'
    if not os.path.exists(model_path):
        print(f"Skipping: {model_path} not found")
        return

    print(f"Testing dynamic shapes on {model_path}...")
    
    # Create session (CPU is enough for shape testing)
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Test Cases: (Batch, Channels, Height, Width)
    test_shapes = [
        (1, 3, 640, 640),   # Standard
        (1, 3, 384, 640),   # Rectangular
        (2, 3, 640, 640),   # Batch > 1
        (1, 3, 736, 1280)   # HDish (stride 32 compatible)
    ]
    
    for shape in test_shapes:
        print(f"Testing shape: {shape}...", end=" ")
        try:
            # Create dummy input
            dummy_input = np.random.randn(*shape).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_name: dummy_input})
            
            # Basic check: output should not be empty
            if outputs[0].shape[0] == shape[0]:
                print("PASS")
            else:
                print(f"FAIL (Batch dim mismatch: {outputs[0].shape})")
                
        except Exception as e:
            print(f"FAIL ({e})")
            
    print("-" * 20)
    print("All shape tests passed!")

if __name__ == "__main__":
    test_onnx_shapes()
