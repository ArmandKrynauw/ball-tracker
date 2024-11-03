import numpy as np
from scipy.spatial.distance import euclidean
from filterpy.kalman import KalmanFilter
from scipy import stats
from sklearn.cluster import DBSCAN

class BallTracker:
    def __init__(self, max_jump_distance=100, max_frames_to_interpolate=20, 
                 lookback_size=5, lookahead_size=5):
        self.max_jump_distance = max_jump_distance
        self.max_frames_to_interpolate = max_frames_to_interpolate
        self.lookback_size = lookback_size
        self.lookahead_size = lookahead_size
        self.init_kalman_filter()

    def init_kalman_filter(self):
        """Initialize Kalman filter for ball tracking"""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, vx, y, vy], Measurement: [x, y]
        dt = 1.0  # time step
        
        # State transition matrix
        self.kf.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 10
        
        # Process noise
        self.kf.Q *= 0.1
        
        # Initial state covariance
        self.kf.P *= 100

    def get_box_center(self, box):
        """Calculate center point of bounding box"""
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    
    def get_box_size(self, box):
        """Calculate width and height of bounding box"""
        return (box[2] - box[0]), (box[3] - box[1])

    def get_box_distance(self, box1, box2):
        """Calculate distance between centers of two bounding boxes"""
        center1 = self.get_box_center(box1)
        center2 = self.get_box_center(box2)
        return euclidean(center1, center2)

    def is_valid_size(self, box, size_stats=None):
        """Check if box size is within reasonable limits"""
        w, h = self.get_box_size(box)
        if size_stats is None:
            # Default size constraints
            return 5 < w < 50 and 5 < h < 50
        
        # Use statistical size constraints
        w_mean, w_std = size_stats['width']
        h_mean, h_std = size_stats['height']
        return (abs(w - w_mean) < 3 * w_std and 
                abs(h - h_mean) < 3 * h_std)

    def calculate_size_statistics(self, predictions):
        """Calculate mean and std of box sizes"""
        widths = []
        heights = []
        
        for pred in predictions.values():
            if len(pred['boxes']) > 0:
                box = pred['boxes'][0]['xyxy']
                w, h = self.get_box_size(box)
                widths.append(w)
                heights.append(h)
        
        if not widths:
            return None
        
        return {
            'width': (np.mean(widths), np.std(widths)),
            'height': (np.mean(heights), np.std(heights))
        }

    def cluster_trajectories(self, valid_detections):
        """Use DBSCAN to cluster trajectories and identify main ball path"""
        if len(valid_detections) < 2:
            return valid_detections
        
        # Extract centers and frame numbers
        centers = np.array([self.get_box_center(box) for _, box in valid_detections])
        frames = np.array([frame for frame, _ in valid_detections]).reshape(-1, 1)
        
        # Combine spatial and temporal information
        features = np.hstack([centers, frames * 0.5])  # Scale frame numbers
        
        # Cluster
        clustering = DBSCAN(eps=50, min_samples=2).fit(features)
        labels = clustering.labels_
        
        # Find main cluster (largest or most continuous)
        if len(set(labels)) > 1:
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            main_label = unique_labels[counts.argmax()]
            main_cluster_indices = np.where(labels == main_label)[0]
            return [valid_detections[i] for i in main_cluster_indices]
        
        return valid_detections

    def predict_with_kalman(self, measurement=None):
        """Predict next position using Kalman filter"""
        if measurement is not None:
            self.kf.predict()
            self.kf.update(measurement)
        else:
            self.kf.predict()
        
        return self.kf.x[:3:2]  # Return predicted [x, y]

    def interpolate_boxes(self, box1, box2, num_steps):
        """Interpolate between two boxes using Kalman predictions"""
        if num_steps <= 0:
            return []
        
        interpolated = []
        center1 = self.get_box_center(box1)
        center2 = self.get_box_center(box2)
        
        # Initialize Kalman filter with first position
        self.kf.x = np.array([center1[0], 0, center1[1], 0])
        
        # Generate interpolated positions
        for i in range(num_steps):
            t = (i + 1) / (num_steps + 1)
            
            # Get Kalman prediction
            predicted_center = self.predict_with_kalman()
            
            # Blend with linear interpolation
            alpha = 0.7  # Weight for Kalman prediction
            blended_center = (alpha * predicted_center + 
                            (1 - alpha) * (center1 + t * (center2 - center1)))
            
            # Calculate box dimensions
            w1, h1 = self.get_box_size(box1)
            w2, h2 = self.get_box_size(box2)
            w = w1 + t * (w2 - w1)
            h = h1 + t * (h2 - h1)
            
            # Create interpolated box
            new_box = np.array([
                blended_center[0] - w/2,
                blended_center[1] - h/2,
                blended_center[0] + w/2,
                blended_center[1] + h/2
            ])
            interpolated.append(new_box)
        
        return interpolated

    def analyze_trajectory_pattern(self, detections, current_idx):
        """
        Analyze local trajectory pattern to determine if a jump is valid
        Returns: bool indicating if the jump should be considered valid
        """
        if len(detections) < 3:
            return False

        # Get surrounding detections
        start_idx = max(0, current_idx - self.lookback_size)
        end_idx = min(len(detections), current_idx + self.lookahead_size + 1)
        
        local_detections = detections[start_idx:end_idx]
        
        if len(local_detections) < 3:
            return False

        # Analyze velocity patterns before and after the jump
        velocities_before = []
        velocities_after = []
        
        for i in range(1, len(local_detections)):
            prev_frame, prev_box = local_detections[i-1]
            curr_frame, curr_box = local_detections[i]
            
            if curr_frame - prev_frame > self.max_frames_to_interpolate:
                continue
                
            velocity = self.get_box_distance(prev_box, curr_box) / (curr_frame - prev_frame)
            
            if i <= current_idx - start_idx:
                velocities_before.append(velocity)
            else:
                velocities_after.append(velocity)

        # If we have enough velocity data, check for consistency
        if velocities_before and velocities_after:
            avg_velocity_before = np.mean(velocities_before)
            avg_velocity_after = np.mean(velocities_after)
            
            # Check if velocities before and after are relatively consistent
            velocity_ratio = min(avg_velocity_before, avg_velocity_after) / max(avg_velocity_before, avg_velocity_after)
            
            # Check if the movement direction is consistent
            if velocity_ratio > 0.3:  # Velocities are somewhat similar
                return True

        return False

    def find_long_range_connections(self, sequences):
        """
        Try to connect sequences that might be part of the same trajectory
        even if they're separated by more frames
        """
        if len(sequences) <= 1:
            return sequences

        connected_sequences = []
        current_sequence = sequences[0]

        for next_sequence in sequences[1:]:
            last_frame, last_box = current_sequence[-1]
            first_frame, first_box = next_sequence[0]
            frame_gap = first_frame - last_frame

            # Check if sequences might be connected despite larger gap
            if frame_gap <= self.max_frames_to_interpolate * 2:  # Allow larger gaps
                distance = self.get_box_distance(last_box, first_box)
                
                # Calculate expected position based on velocity
                if len(current_sequence) >= 2:
                    prev_frame, prev_box = current_sequence[-2]
                    velocity = self.get_box_distance(prev_box, last_box) / (last_frame - prev_frame)
                    expected_distance = velocity * frame_gap
                    
                    # If the distance is reasonable given the velocity
                    if distance < expected_distance * 1.5:
                        current_sequence.extend(next_sequence)
                        continue

            connected_sequences.append(current_sequence)
            current_sequence = next_sequence

        connected_sequences.append(current_sequence)
        return connected_sequences

    def find_valid_sequences(self, predictions):
        """Find sequences of valid ball positions with improved jump detection"""
        size_stats = self.calculate_size_statistics(predictions)
        
        # Collect all valid detections
        valid_detections = []
        for frame_idx, pred in sorted(predictions.items()):
            if len(pred['boxes']) > 0:
                box = pred['boxes'][0]['xyxy']
                if self.is_valid_size(box, size_stats):
                    valid_detections.append((frame_idx, box))

        # Cluster trajectories to identify main ball path
        valid_detections = self.cluster_trajectories(valid_detections)
        
        # Split into initial sequences
        sequences = []
        current_sequence = []
        
        for i in range(len(valid_detections)):
            frame_idx, box = valid_detections[i]
            
            if not current_sequence:
                current_sequence.append((frame_idx, box))
            else:
                last_frame, last_box = current_sequence[-1]
                frame_gap = frame_idx - last_frame
                distance = self.get_box_distance(box, last_box)
                
                # Check if this is a valid jump using trajectory analysis
                is_valid_jump = self.analyze_trajectory_pattern(valid_detections, i)
                
                if (frame_gap <= self.max_frames_to_interpolate or 
                    distance <= self.max_jump_distance or 
                    is_valid_jump):
                    current_sequence.append((frame_idx, box))
                else:
                    if len(current_sequence) > 1:
                        sequences.append(current_sequence)
                    current_sequence = [(frame_idx, box)]
        
        if len(current_sequence) > 1:
            sequences.append(current_sequence)

        # Try to connect sequences that might be part of the same trajectory
        sequences = self.find_long_range_connections(sequences)
        
        return sequences

    def process_predictions(self, predictions):
        """Process predictions with improved interpolation"""
        filtered_predictions = {k: {'boxes': []} for k in predictions.keys()}
        valid_sequences = self.find_valid_sequences(predictions)
        
        for sequence in valid_sequences:
            # Process each consecutive pair in the sequence
            for i in range(len(sequence) - 1):
                frame1, box1 = sequence[i]
                frame2, box2 = sequence[i + 1]
                frame_gap = frame2 - frame1
                
                # Add first frame
                filtered_predictions[frame1] = {
                    'boxes': [{'xyxy': np.array(box1)}]
                }
                
                # Interpolate if there's a gap
                if frame_gap > 1:
                    # Use velocity-aware interpolation for longer gaps
                    if frame_gap > self.max_frames_to_interpolate:
                        # Calculate velocity from previous frames if available
                        if i > 0:
                            prev_frame, prev_box = sequence[i-1]
                            velocity_vector = (self.get_box_center(box1) - 
                                            self.get_box_center(prev_box)) / (frame1 - prev_frame)
                        else:
                            velocity_vector = (self.get_box_center(box2) - 
                                            self.get_box_center(box1)) / frame_gap

                        # Use velocity-based prediction for initial interpolation
                        interpolated = self.interpolate_boxes(box1, box2, frame_gap - 1)
                    else:
                        interpolated = self.interpolate_boxes(box1, box2, frame_gap - 1)
                    
                    for idx, box in enumerate(interpolated, 1):
                        frame_idx = frame1 + idx
                        filtered_predictions[frame_idx] = {
                            'boxes': [{'xyxy': box}]
                        }
            
            # Add last frame of sequence
            last_frame, last_box = sequence[-1]
            filtered_predictions[last_frame] = {
                'boxes': [{'xyxy': np.array(last_box)}]
            }
        
        return filtered_predictions

def filter_and_interpolate_predictions(predictions, image_paths, 
                                    max_distance=100, max_gap=20,
                                    lookback=5, lookahead=5):
    """Main function to filter and interpolate predictions"""
    tracker = BallTracker(
        max_jump_distance=max_distance,
        max_frames_to_interpolate=max_gap,
        lookback_size=lookback,
        lookahead_size=lookahead
    )
    return tracker.process_predictions(predictions)