import pytest
import numpy as np
from inference.tracker import ByteTracker
from inference.fusion import FusionEngine

def test_tracker_initialization():
    tracker = ByteTracker()
    assert len(tracker.tracked_stracks) == 0
    assert tracker.frame_id == 0

def test_iou_calculation():
    tracker = ByteTracker()
    # Box: [x1, y1, x2, y2]
    # Perfect overlap
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    iou = tracker.compute_iou(box1, box2)
    assert abs(iou - 1.0) < 1e-5
    
    # No overlap
    box3 = [20, 20, 30, 30]
    iou_none = tracker.compute_iou(box1, box3)
    assert iou_none == 0.0
    
    # Partial overlap
    # box1 area = 100. box4: 5,0,15,10 -> intersect [5,0,10,10], w=5 h=10 area=50.
    # Union = 100+100-50=150. IoU=0.33
    box4 = [5, 0, 15, 10] 
    iou_part = tracker.compute_iou(box1, box4)
    assert 0.3 < iou_part < 0.35

def test_new_track_creation():
    tracker = ByteTracker(track_thresh=0.5)
    # [x1, y1, x2, y2, score, class]
    dets = np.array([[10, 10, 60, 60, 0.9, 0]])
    frame = None
    
    tracks = tracker.update(dets, frame)
    
    assert len(tracks) == 1
    assert tracks[0][4] == 1 # ID 1

def test_track_continuation():
    tracker = ByteTracker()
    dets1 = np.array([[10, 10, 60, 60, 0.9, 0]])
    tracks1 = tracker.update(dets1, None)
    id1 = tracks1[0][4]
    
    # Move slightly (Kalman filter should match)
    dets2 = np.array([[12, 12, 62, 62, 0.9, 0]])
    tracks2 = tracker.update(dets2, None)
    id2 = tracks2[0][4]
    
    assert id1 == id2

# --- MERGED TESTS: Tracker Drift ---

def test_drift_logic():
    fusion = FusionEngine(drift_threshold=0.5)
    tracks = [[100, 100, 50, 50]]
    
    # Case 1: Perfect Overlap (No Drift)
    dets_ok = [[100, 100, 150, 150]]
    assert fusion.check_drift(dets_ok, tracks) is False
    
    # Case 2: No Overlap (Drift)
    dets_bad = [[200, 200, 250, 250]]
    assert fusion.check_drift(dets_bad, tracks) is True
    
    # Case 3: Partial Overlap (IoU < 0.5 -> Drift)
    dets_partial = [[140, 100, 190, 150]] # Sligthly shifted
    assert fusion.check_drift(dets_partial, tracks) is True
