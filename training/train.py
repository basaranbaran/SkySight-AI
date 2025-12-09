import argparse
import os
from ultralytics import YOLO
import torch

def train_model(
    data_cfg: str, 
    epochs: int = 50, 
    imgsz: int = 640, 
    batch: int = 16, 
    model_name: str = 'yolov8s.pt',
    project_dir: str = 'runs/detect',
    name: str = 'visdrone_v1',
    workers: int = 8
):
    print(f"Starting Production Training with model: {model_name}")
    print("Configuration: AMP=True, CosineLR=True, Augments=Strong")
    
    # Initialize YOLOv8 model
    model = YOLO(model_name)

    # Train the model with VisDrone specific hyperparams
    results = model.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,    # Limit dataloader workers
        # Optimization & Schedulers
        amp=True,           # Automatic Mixed Precision
        cos_lr=True,        # Cosine LR scheduler
        optimizer='AdamW',  # Optimizer
        
        # Explicit Requirements
        ema=True,           # Exponential Moving Average (Mandatory)
        multi_scale=True,   # Multi-scale training (Mandatory)
        rect=False,         # Disable rectangular training to allow multi-scale mosaic
        
        # Logging
        project='training/logs',
        name=name,
        
        # Hyperparameters (overrides)
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        lr0=0.002,          # Initial LR
        lrf=0.01,           # Final LR fraction
        
        # Augments
        mosaic=1.0,         
        mixup=0.15,         
        copy_paste=0.1,     
        
        degrees=10.0,       
        translate=0.1,      
        scale=0.5,          
        shear=2.0,          
        perspective=0.0005, 
        flipud=0.0,         
        fliplr=0.5,         
        
        # Color Jitter
        hsv_h=0.015,        
        hsv_s=0.7,          
        hsv_v=0.4,          
        
        # Settings
        exist_ok=True,      
        plots=True,         
        save=True,          
    )

    # Export the best model
    # Note: project set to 'training/logs', name set to arg.name
    best_model_path = os.path.join('training', 'logs', name, 'weights', 'best.pt')
    target_path = os.path.join('models', 'visdrone_v8s.pt')
    
    if os.path.exists(best_model_path):
        os.makedirs('models', exist_ok=True)
        import shutil
        target_name = f"{name}.pt"
        target_path = os.path.join('models', target_name)
        shutil.copy(best_model_path, target_path)
        print(f"Dataset Training complete. Best model saved to {target_path}")
    else:
        print("Training finished but could not find best.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VisDrone model')
    parser.add_argument('--data', type=str, default='VisDrone.yaml', help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size') # Lower batch for 8s on 3060
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Model architecture')
    parser.add_argument('--name', type=str, default='visdrone_v1', help='Experiment name')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    # Ensure W&B is login if needed or disabled, user can configure env vars
    # os.environ['WANDB_MODE'] = 'offline' # Uncomment to disable online logging
    
    train_model(
        data_cfg=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
        name=args.name,
        workers=args.workers
    )
