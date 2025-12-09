import numpy as np

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou

class FusionEngine:
    def __init__(self, drift_threshold=0.5):
        self.drift_threshold = drift_threshold
        
    def check_drift(self, det_boxes, track_boxes):
        """
        Check if tracker has drifted significantly from new detections.
        Simple logic: For each track, find best matching detection. If IoU < threshold, flag drift.
        """
        # This is a simplified drift check. 
        # In a real scenario, you'd match IDs if possible or just use location.
        # Here we just return a boolean if global drift is suspected or if we should trust detections more.
        
        if len(det_boxes) == 0 or len(track_boxes) == 0:
             return False # Cannot compare
             
        # Simple heuristic: Compute mean IoU of best matches
        total_iou = 0
        matches = 0
        
        for trk in track_boxes:
            # trk: x, y, w, h -> x1, y1, x2, y2
            t_box = [trk[0], trk[1], trk[0]+trk[2], trk[1]+trk[3]]
            
            best_iou = 0
            for det in det_boxes:
                # det: x1, y1, x2, y2 ...
                d_box = det[:4]
                iou = compute_iou(t_box, d_box)
                if iou > best_iou:
                    best_iou = iou
            
            if best_iou > 0:
                total_iou += best_iou
                matches += 1
                
        if matches == 0:
            return True # Complete drift (no overlaps)
            
        avg_iou = total_iou / matches
        return avg_iou < self.drift_threshold
