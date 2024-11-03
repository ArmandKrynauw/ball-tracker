import numpy as np
from numpy.random import randn
from scipy.spatial.distance import euclidean

class ParticleFilter:
    def __init__(self, n_particles=100, process_std=5.0, measurement_std=10.0):
        """
        Initialize the particle filter.

        Args:
            n_particles (int): Number of particles to use
            process_std (float): Standard deviation of process noise
            measurement_std (float): Standard deviation of measurement noise
        """
        self.n_particles = n_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.particles = None
        self.weights = np.ones(n_particles) / n_particles
        self.initialized = False

    def initialize(self, initial_position):
        """Initialize particles around first detection."""
        self.particles = np.repeat(initial_position.reshape(1, -1), self.n_particles, axis=0)
        self.particles += randn(self.n_particles, 2) * self.measurement_std
        self.initialized = True

    def predict(self):
        """Predict next state using particle motion model."""
        if not self.initialized:
            return None

        # Add random noise to particles (process noise)
        self.particles += randn(*self.particles.shape) * self.process_std

        # Return weighted mean of particles as prediction
        return np.average(self.particles, weights=self.weights, axis=0)

    def update(self, measurement):
        """Update particle weights based on measurement."""
        if measurement is None:
            return self.predict()

        if not self.initialized:
            self.initialize(measurement)
            return measurement

        # Calculate weights based on measurement likelihood
        dist = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights *= np.exp(-0.5 * (dist ** 2) / (self.measurement_std ** 2))
        self.weights += 1.e-300  # Avoid numerical underflow
        self.weights /= sum(self.weights)  # Normalize weights

        # Resample particles if effective sample size is too low
        if self._neff() < self.n_particles / 2:
            self._resample()

        # Return weighted mean of particles
        return np.average(self.particles, weights=self.weights, axis=0)

    def _neff(self):
        """Calculate effective sample size."""
        return 1. / np.sum(self.weights ** 2)

    def _resample(self):
        """Resample particles based on their weights."""
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.
        indices = np.searchsorted(cumsum, np.random.random(self.n_particles))
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

def filter_and_interpolate_predictions(predictions, max_gap=5, max_distance=50):
    """
    Filters and interpolates the predictions using a particle filter to
    smooth trajectories and fill in missing detections.

    Args:
        predictions (dict): Original predictions with frame indices as keys
        max_gap (int): Maximum gap of frames to interpolate between
        max_distance (float): Maximum distance to consider a detection valid

    Returns:
        dict: New set of predictions with filtered and interpolated positions
    """
    pf = ParticleFilter(n_particles=100, process_std=5.0, measurement_std=10.0)
    filtered_predictions = {}
    previous_position = None

    for frame_index in sorted(predictions.keys()):
        # Check if a ball was detected in this frame
        if predictions[frame_index].boxes:
            box = predictions[frame_index].boxes[0].cpu()
            x1, y1, x2, y2 = box.xyxy[0]
            ball_position = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Center of bounding box

            # Discard detections too far from the previous position
            if previous_position is not None and euclidean(ball_position, previous_position) > max_distance:
                ball_position = None
            else:
                # Update particle filter with the detected position
                filtered_position = pf.update(ball_position)
                previous_position = filtered_position
        else:
            ball_position = None

        # If no ball detected, use prediction from particle filter
        if ball_position is None:
            if previous_position is not None:
                # Predict next position using particle filter
                filtered_position = pf.predict()
                if filtered_position is None:
                    filtered_predictions[frame_index] = {'boxes': []}
                    continue
            else:
                # Skip if there's no prior position for initialization
                filtered_predictions[frame_index] = {'boxes': []}
                continue

        # Convert position back to bbox format
        box_size = 20  # Adjust this value based on your typical ball size
        filtered_predictions[frame_index] = {
            'boxes': [{
                'xyxy': np.array([
                    filtered_position[0] - box_size/2,
                    filtered_position[1] - box_size/2,
                    filtered_position[0] + box_size/2,
                    filtered_position[1] + box_size/2
                ])
            }]
        }

    # Fill in gaps using linear interpolation
    frame_indices = sorted(filtered_predictions.keys())
    for i in range(len(frame_indices) - 1):
        start_frame = frame_indices[i]
        end_frame = frame_indices[i + 1]
        gap = end_frame - start_frame

        if gap > 1 and gap <= max_gap:
            if (len(filtered_predictions[start_frame]['boxes']) > 0 and 
                len(filtered_predictions[end_frame]['boxes']) > 0):
                # Get start and end positions
                start_box = filtered_predictions[start_frame]['boxes'][0]['xyxy']
                end_box = filtered_predictions[end_frame]['boxes'][0]['xyxy']
                start_pos = np.array([(start_box[0] + start_box[2])/2, 
                                    (start_box[1] + start_box[3])/2])
                end_pos = np.array([(end_box[0] + end_box[2])/2, 
                                  (end_box[1] + end_box[3])/2])

                # Interpolate between frames
                for j in range(1, gap):
                    t = j / gap
                    interp_pos = start_pos + (end_pos - start_pos) * t
                    filtered_predictions[start_frame + j] = {
                        'boxes': [{
                            'xyxy': np.array([
                                interp_pos[0] - box_size/2,
                                interp_pos[1] - box_size/2,
                                interp_pos[0] + box_size/2,
                                interp_pos[1] + box_size/2
                            ])
                        }]
                    }

    return filtered_predictions
