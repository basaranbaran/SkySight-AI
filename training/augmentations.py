import albumentations as A
import cv2
import numpy as np

def get_train_transforms(image_size=640):
    """
    Returns a strong augmentation pipeline using Albumentations.
    Includes:
    - Mosaic (conceptually, though usually handled by YOLO dataloader, adding spatial augs here)
    - MotionBlur
    - RandomCrop / RandomResizedCrop
    - ColorJitter (HueSaturationValue, RandomBrightnessContrast)
    - Cutout (CoarseDropout)
    """
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=0.5),
        A.HorizontalFlip(p=0.5),
        
        # Color & Blur augmentations
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        
        A.HueSaturationValue(p=0.3),
        
        # Cutout / CoarseDropout
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32, 
            min_holes=1, min_height=16, min_width=16, 
            fill_value=0, mask_fill_value=0, p=0.2
        ),
        
        # Note: Normalize and ToTensor are typically handled by the YOLOv8 internal pipeline
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

class AlbumentationsWrapper:
    """
    Wrapper to integrate Albumentations with YOLOv8's training loop if needed,
    or simply to be used as a reference for pre-processing.
    YOLOv8 has built-in support for Albumentations, but this defines our custom policy.
    """
    def __init__(self, image_size=640):
        self.transform = get_train_transforms(image_size)

    def __call__(self, image, bboxes, class_labels):
        annotated = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return annotated['image'], annotated['bboxes'], annotated['class_labels']
