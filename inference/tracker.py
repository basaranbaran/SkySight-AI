import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    def __init__(self):
        # 8x1 state vector [cx, cy, aspect_ratio, height, vx, vy, va, vh]
        self.mean = np.zeros(8)
        self.covariance = np.eye(8)
        
        # Motion model (constant velocity)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1.0
            
        # Measurement matrix (only observe first 4 items)
        self.H = np.eye(4, 8)
        
        self.R = np.eye(4) # Measurement noise
        self.P = np.eye(8) * 10
        self.Q = np.eye(8) * 0.01 # Process noise

    def predict(self):
        self.mean = np.dot(self.F, self.mean)
        self.covariance = np.linalg.multi_dot((self.F, self.covariance, self.F.T)) + self.Q
        return self.mean, self.covariance

    def update(self, measurement):
        # measurement: [cx, cy, a, h]
        y = measurement - np.dot(self.H, self.mean)
        S = np.linalg.multi_dot((self.H, self.covariance, self.H.T)) + self.R
        K = np.linalg.multi_dot((self.covariance, self.H.T, np.linalg.inv(S)))
        
        self.mean = self.mean + np.dot(K, y)
        I = np.eye(8)
        self.covariance = np.dot(I - np.dot(K, self.H), self.covariance)

class Track:
    def __init__(self, tlwh, score, cls_id, track_id):
        self.track_id = track_id
        self.score = score
        self.cls_id = cls_id
        self.state = 1 # 1=New, 2=Tracked, 3=Lost
        
        # Convert tlwh to [cx, cy, a, h]
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.mean, self.kf.covariance
        self.init_kf(tlwh)
        
        self.frame_id = 0
        self.tracklet_len = 0
        self.lost_frames = 0
        
    def init_kf(self, tlwh):
        cx = tlwh[0] + tlwh[2]/2
        cy = tlwh[1] + tlwh[3]/2
        a = tlwh[2] / tlwh[3]
        h = tlwh[3]
        self.kf.mean[:4] = [cx, cy, a, h]
        self.kf.covariance[:4, :4] *= 10

    def predict(self):
        self.kf.predict()
        
    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # Update KF
        tlwh = new_track.tlwh
        cx = tlwh[0] + tlwh[2]/2
        cy = tlwh[1] + tlwh[3]/2
        a = tlwh[2] / tlwh[3]
        h = tlwh[3]
        self.kf.update(np.array([cx, cy, a, h]))
        
        self.score = new_track.score
        self.state = 2 # Tracked
        self.lost_frames = 0

    @property
    def tlwh(self):
        # Convert state [cx, cy, a, h] back to tlwh
        cx, cy, a, h = self.kf.mean[:4]
        w = h * a
        return np.array([cx - w/2, cy - h/2, w, h])

class ByteTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        self.track_id_count = 0

    def update(self, dets, img=None):
        self.frame_id += 1
        
        # Pre-process detections
        if hasattr(dets, 'cpu'): dets = dets.cpu().numpy()
        
        scores = dets[:, 4]
        bboxes = dets[:, :4]
        classes = dets[:, 5]
        
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        
        inds_second = np.logical_and(inds_low, inds_high)
        
        dets_first = dets[remain_inds]
        dets_second = dets[inds_second]
        
        # Create 'Track' objects for detections (not yet IDed)
        strack_pool = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
        
        # Predict KF for existing tracks
        for strack in strack_pool:
            strack.predict()
            
        # Association 1: High Confidence Detections
        dists = self.iou_distance(strack_pool, dets_first)
        matches, u_track, u_detection = self.linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = self.transform_det(dets_first[idet])
            if track.state == 2:
                track.update(det, self.frame_id)
                self.tracked_stracks.append(track) # Re-add updated tracks? No, manage lists better
            else:
                track.update(det, self.frame_id)
                self.tracked_stracks.append(track)

        # Simplified List Management for this quick implementation:
        # Re-build lists every frame is easier for logic clarity
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Need to handle matches correctly
        # Re-doing list logic properly:
        
        current_tracked = []
        
        # --- STAGE 1 Matches ---
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det_obj = self.transform_det(dets_first[idet])
            track.update(det_obj, self.frame_id)
            current_tracked.append(track)
            
        # --- STAGE 2: Low Confidence Matches (ByteTrack core) ---
        # Candidates are tracks that were NOT matched in Stage 1
        # Only consider currently tracked tracks for this stage usually, or all unmatch?
        # ByteTrack uses 'r_tracked_stracks' (running tracks) usually.
        
        # Simplified: unmatched high-conf tracks + low-conf detections
        candidates = [strack_pool[i] for i in u_track if strack_pool[i].state == 2] # Only active tracks
        
        dists = self.iou_distance(candidates, dets_second)
        matches_2, u_track_2, u_detection_2 = self.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches_2:
            track = candidates[itracked]
            det_obj = self.transform_det(dets_second[idet])
            track.update(det_obj, self.frame_id)
            current_tracked.append(track)
            
        # --- STAGE 3: Unmatched High Conf Detections -> New Tracks ---
        for idet in u_detection:
            det_obj = self.transform_det(dets_first[idet])
            if det_obj.score > self.track_thresh:
                self.track_id_count += 1
                det_obj.track_id = self.track_id_count
                det_obj.state = 2
                current_tracked.append(det_obj)
        
        # --- STAGE 4: Handle Lost ---
        # Tracks not matched in Stage 1 or 2
        
        # Unmatched from Stage 2 (original active tracks)
        for i in u_track_2:
            track = candidates[i]
            track.lost_frames += 1
            if track.lost_frames > self.track_buffer:
                track.state = 3 # Removed
            else:
                track.state = 3 # Lost
                lost_stracks.append(track)

        # Unmatched from Stage 1 that were 'Lost' state originally
        for i in u_track:
            track = strack_pool[i]
            if track not in candidates: # These were already 'Lost' tracks
                track.lost_frames += 1
                if track.lost_frames > self.track_buffer:
                    track.state = 3
                else:
                    lost_stracks.append(track)

        self.tracked_stracks = current_tracked
        self.lost_stracks = lost_stracks
        
        # Output format [x, y, w, h, id]
        results = []
        for t in self.tracked_stracks:
            if t.state == 2:
                tlwh = t.tlwh
                results.append([tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.track_id])
                
        return results

    def joint_stracks(self, tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            if t.track_id not in exists:
                res.append(t)
        return res

    def transform_det(self, det):
        # det: x1, y1, x2, y2, score, cls
        x1, y1, x2, y2 = det[:4]
        w = x2 - x1
        h = y2 - y1
        score = det[4]
        cls_id = int(det[5])
        return Track(np.array([x1, y1, w, h]), score, cls_id, 0)

    def iou_distance(self, atracks, btracks):
        # atracks: list of Track, btracks: numpy array of dets OR list of Track
        if len(atracks) == 0 or len(btracks) == 0:
            return np.zeros((len(atracks), len(btracks)))
            
        ious = np.zeros((len(atracks), len(btracks)))
        for i, t in enumerate(atracks):
            for j, d in enumerate(btracks):
                # if btracks is raw dets
                if isinstance(d, np.ndarray): 
                     # x1, y1, x2, y2
                     box_d = d[:4]
                     # track to box
                     tlwh = t.tlwh
                     box_t = [tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]]
                else:
                     # d is Track object? No, usually raw det in this logic
                     pass

                ious[i, j] = self.compute_iou(box_t, box_d)
        return 1 - ious # Cost matrix

    def compute_iou(self, box1, box2):
        # x1, y1, x2, y2
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
            
        matches = []
        u_track = []
        u_detection = []
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
            else:
                u_track.append(r)
                u_detection.append(c)
                
        # Add unmatched that linear_assignment didn't pick
        matched_rows = set(r for r,c in matches)
        matched_cols = set(c for r,c in matches)
        
        # Add strict unmatched (above thresh) + ignored
        for r in range(cost_matrix.shape[0]):
            if r not in matched_rows and r not in u_track:
                u_track.append(r)
                
        for c in range(cost_matrix.shape[1]):
            if c not in matched_cols and c not in u_detection:
                u_detection.append(c)
                
        return matches, u_track, u_detection

def get_tracker(name='bytetrack'):
    return ByteTracker()
