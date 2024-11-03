import numpy as np
from numpy.random import randn
from scipy.spatial.distance import euclidean

class ParticleFilter:
    def __init__(self, n_particles=200, process_std=3.0, measurement_std=5.0):
        """
        Initialize the particle filter.
        
        Args:
            n_particles (int): Number of particles
            process_std (float): Process noise standard deviation
            measurement_std (float): Measurement noise standard deviation
        """
        self.n_particles = n_particles
        self.process_std = process_std
        self.measurement_std = measurement_std
        self.particles = None
        self.velocities = None
        self.weights = np.ones(n_particles) / n_particles
        self.initialized = False
        self.prev_position = None
        self.dt = 1.0  # time step
        self.velocity_decay = 0.95  # velocity decay factor

    def initialize(self, initial_position):
        """Initialize particle filter with first detection."""
        self.particles = np.repeat(initial_position.reshape(1, -1), self.n_particles, axis=0)
        self.velocities = np.zeros((self.n_particles, 2))
        self.particles += randn(self.n_particles, 2) * self.measurement_std * 0.1
        self.prev_position = initial_position
        self.initialized = True

    def predict(self):
        """Predict next state using particle motion model."""
        if not self.initialized:
            return None

        # Update positions based on velocity
        self.particles += self.velocities * self.dt
        
        # Add random noise to both position and velocity
        self.particles += randn(*self.particles.shape) * self.process_std
        self.velocities += randn(*self.velocities.shape) * self.process_std * 0.5
        
        # Apply velocity decay (drag)
        self.velocities *= self.velocity_decay
        
        # Return weighted mean of particles
        return np.average(self.particles, weights=self.weights, axis=0)

    def update(self, measurement):
        """Update particle weights based on measurement."""
        if measurement is None:
            return self.predict()

        if not self.initialized:
            self.initialize(measurement)
            return measurement

        # Predict step
        predicted_pos = self.predict()

        if predicted_pos is None:
            return None

        # Calculate distances for all particles
        dist = np.linalg.norm(self.particles - measurement, axis=1)
        
        # Exponential weight update with adaptive scaling
        max_dist = max(dist.max(), 1e-6)
        scaled_dist = dist / max_dist
        self.weights *= np.exp(-0.5 * (scaled_dist ** 2))
        
        # Avoid numerical underflow
        self.weights += 1e-300
        self.weights /= self.weights.sum()

        # Update velocities based on measurement
        measured_velocity = (measurement - self.prev_position) / self.dt
        self.velocities = 0.7 * self.velocities + 0.3 * measured_velocity
        
        # Store current position for next velocity calculation
        self.prev_position = measurement.copy()

        # Resample if effective sample size is too low
        if self._neff() < self.n_particles / 1.5:
            self._resample()

        # Return weighted mean of particles
        return np.average(self.particles, weights=self.weights, axis=0)

    def _neff(self):
        """Calculate effective sample size."""
        return 1.0 / np.sum(np.square(self.weights))

    def _resample(self):
        """Systematic resampling of particles."""
        # Systematic resampling
        positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # Handle numerical errors
        
        indices = np.searchsorted(cumsum, positions)
        
        # Copy with noise to avoid particle depletion
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        
        # Add small random noise to resampled particles
        self.particles += randn(*self.particles.shape) * self.measurement_std * 0.1
        self.velocities += randn(*self.velocities.shape) * self.process_std * 0.1
        
        # Reset weights
        self.weights = np.ones(self.n_particles) / self.n_particles

def filter_and_interpolate_predictions(predictions, max_gap=5, max_distance=50):
    """
    Filters and interpolates predictions using particle filter.
    
    Args:
        predictions (dict): Original predictions with frame indices as keys
        max_gap (int): Maximum frames gap to interpolate
        max_distance (float): Maximum distance for valid detection
    
    Returns:
        dict: Filtered and interpolated predictions
    """
    pf = ParticleFilter(n_particles=200, process_std=3.0, measurement_std=5.0)
    filtered_predictions = {}
    previous_position = None
    min_box_size = 15
    max_box_size = 25

    for frame_index in sorted(predictions.keys()):
        # Get detection for current frame
        current_pred = predictions[frame_index]
        
        if current_pred['boxes']:
            # Extract center position from detection
            box = current_pred['boxes'][0]['xyxy']
            ball_position = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

            # Validate detection using distance threshold
            if previous_position is not None:
                if euclidean(ball_position, previous_position) > max_distance:
                    ball_position = None
            
            if ball_position is not None:
                # Update filter with valid detection
                filtered_position = pf.update(ball_position)
                previous_position = filtered_position
        else:
            # No detection in this frame
            ball_position = None
            filtered_position = pf.predict() if pf.initialized else None

        # Store filtered prediction
        if filtered_position is not None:
            # Calculate adaptive box size based on velocity
            if pf.velocities is not None:
                velocity_magnitude = np.linalg.norm(pf.velocities.mean(axis=0))
                box_size = max(min_box_size, 
                             min(max_box_size, min_box_size + velocity_magnitude * 0.5))
            else:
                box_size = min_box_size

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
        else:
            filtered_predictions[frame_index] = {'boxes': []}

    # Fill gaps using linear interpolation
    frame_indices = sorted(filtered_predictions.keys())
    for i in range(len(frame_indices) - 1):
        start_frame = frame_indices[i]
        end_frame = frame_indices[i + 1]
        gap = end_frame - start_frame

        if gap > 1 and gap <= max_gap:
            if (filtered_predictions[start_frame]['boxes'] and 
                filtered_predictions[end_frame]['boxes']):
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
                    
                    # Use average box size for interpolated frames
                    box_size = (min_box_size + max_box_size) / 2
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
