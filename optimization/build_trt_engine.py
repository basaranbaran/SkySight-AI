import argparse
import os
import tensorrt as trt
import sys

# Logger is required for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path, precision='fp16', calibration_cache=None, calib_data_path=None):
    """
    Builds a TensorRT engine from an ONNX file.
    """
    print(f"Building {precision} engine from {onnx_file_path}...")
    
    builder = trt.Builder(TRT_LOGGER)
    
    # Create network definition with explicit batch
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # Parse ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file {onnx_file_path} not found.")
        return

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Memory configuration (max workspace size)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Precision configuration
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: FP16 not supported on this platform.")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calib_data_path:
                from calibrate_int8 import YOLOEntropyCalibrator
                config.int8_calibrator = YOLOEntropyCalibrator(calib_data_path, cache_file=calibration_cache)
            else:
                 print("Error: INT8 precision requires calibration data path.")
                 return
        else:
             print("Warning: INT8 not supported on this platform.")

    # Optimization Profiles (Dynamic Shapes)
    profile = builder.create_optimization_profile()
    # Input name usually 'images' for YOLO, but check your ONNX export
    input_name = network.get_input(0).name
    print(f"Input tensor name: {input_name}")
    
    # Min, Opt, Max shapes: (Batch, Channels, Height, Width)
    # Min, Opt, Max shapes: (Batch, Channels, Height, Width)
    # Updated for variable resolution support (Min 640 -> Max 1280)
    profile.set_shape(input_name, (1, 3, 640, 640), (2, 3, 960, 960), (4, 3, 1280, 1280))
    config.add_optimization_profile(profile)

    # Build engine
    print("Building engine (this may take a while)...")
    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError: 
        # For older TRT versions
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize() if engine else None

    if engine_bytes:
        with open(engine_file_path, "wb") as f:
            f.write(engine_bytes)
        print(f"Engine saved to {engine_file_path}")
    else:
        print("Failed to build engine.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--modelfile', type=str, default='', help='Output engine path (optional)')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'], help='Precision')
    parser.add_argument('--calib-data', type=str, default=None, help='Path to calibration images directory')
    args = parser.parse_args()

    output_path = args.modelfile if args.modelfile else args.onnx.replace('.onnx', f'_{args.precision}.engine')
    cache_path = 'models/calibration.cache'
    
    build_engine(args.onnx, output_path, args.precision, cache_path, args.calib_data)
