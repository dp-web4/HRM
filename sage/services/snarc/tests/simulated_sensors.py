"""
Simulated Sensor Streams for SNARC Testing

Generates realistic sensor data patterns for testing salience assessment:
- Periodic signals (predictable, low surprise)
- Step changes (high surprise at transition)
- Random noise (high novelty)
- Anomalies (high arousal, conflict)
- Goal-relevant patterns (high reward)
"""

import numpy as np
import torch
from typing import Dict, Any, Callable, Optional
import time


class SensorSimulator:
    """Base class for sensor simulators"""

    def __init__(self, sensor_id: str, sample_rate: float = 10.0):
        """
        Args:
            sensor_id: Unique sensor identifier
            sample_rate: Samples per second
        """
        self.sensor_id = sensor_id
        self.sample_rate = sample_rate
        self.time_step = 0
        self.start_time = time.time()

    def get_sample(self) -> Any:
        """Generate next sensor sample"""
        raise NotImplementedError

    def reset(self):
        """Reset sensor to initial state"""
        self.time_step = 0
        self.start_time = time.time()


class PeriodicSensor(SensorSimulator):
    """
    Generates periodic signal (sine wave)

    Characteristics:
    - Low surprise (predictable)
    - Low novelty (repeating pattern)
    - Moderate arousal (amplitude)
    - Low conflict
    """

    def __init__(
        self,
        sensor_id: str,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        noise_level: float = 0.1,
        sample_rate: float = 10.0
    ):
        super().__init__(sensor_id, sample_rate)
        self.frequency = frequency
        self.amplitude = amplitude
        self.noise_level = noise_level

    def get_sample(self) -> float:
        """Generate periodic value with optional noise"""
        t = self.time_step / self.sample_rate
        value = self.amplitude * np.sin(2 * np.pi * self.frequency * t)

        if self.noise_level > 0:
            value += np.random.normal(0, self.noise_level)

        self.time_step += 1
        return float(value)


class StepChangeSensor(SensorSimulator):
    """
    Generates signal with sudden step changes

    Characteristics:
    - High surprise at transitions
    - Moderate novelty (new level)
    - High arousal (sudden change)
    - Potential conflict (if other sensors don't change)
    """

    def __init__(
        self,
        sensor_id: str,
        baseline: float = 0.0,
        step_value: float = 5.0,
        step_interval: int = 50,  # steps between changes
        noise_level: float = 0.1,
        sample_rate: float = 10.0
    ):
        super().__init__(sensor_id, sample_rate)
        self.baseline = baseline
        self.step_value = step_value
        self.step_interval = step_interval
        self.noise_level = noise_level
        self.current_level = baseline

    def get_sample(self) -> float:
        """Generate value with step changes"""
        # Check if it's time for a step change
        if self.time_step > 0 and self.time_step % self.step_interval == 0:
            # Toggle between baseline and step_value
            if self.current_level == self.baseline:
                self.current_level = self.step_value
            else:
                self.current_level = self.baseline

        value = self.current_level

        if self.noise_level > 0:
            value += np.random.normal(0, self.noise_level)

        self.time_step += 1
        return float(value)


class RandomWalkSensor(SensorSimulator):
    """
    Generates random walk (Brownian motion)

    Characteristics:
    - Moderate surprise (somewhat predictable)
    - High novelty (always exploring new values)
    - Variable arousal (magnitude changes)
    - Low conflict (smooth changes)
    """

    def __init__(
        self,
        sensor_id: str,
        step_size: float = 0.5,
        bounds: Optional[tuple] = (-10.0, 10.0),
        sample_rate: float = 10.0
    ):
        super().__init__(sensor_id, sample_rate)
        self.step_size = step_size
        self.bounds = bounds
        self.current_value = 0.0

    def get_sample(self) -> float:
        """Generate random walk step"""
        # Take random step
        self.current_value += np.random.normal(0, self.step_size)

        # Apply bounds if specified
        if self.bounds is not None:
            self.current_value = np.clip(
                self.current_value,
                self.bounds[0],
                self.bounds[1]
            )

        self.time_step += 1
        return float(self.current_value)


class AnomalySensor(SensorSimulator):
    """
    Generates normal signal with occasional anomalies

    Characteristics:
    - Very high surprise at anomaly
    - Very high novelty (rare pattern)
    - Very high arousal (large magnitude)
    - High conflict (disagrees with other sensors)
    """

    def __init__(
        self,
        sensor_id: str,
        baseline_mean: float = 0.0,
        baseline_std: float = 1.0,
        anomaly_probability: float = 0.05,
        anomaly_magnitude: float = 10.0,
        sample_rate: float = 10.0
    ):
        super().__init__(sensor_id, sample_rate)
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        self.anomaly_probability = anomaly_probability
        self.anomaly_magnitude = anomaly_magnitude

    def get_sample(self) -> float:
        """Generate normal value or anomaly"""
        # Randomly trigger anomaly
        if np.random.random() < self.anomaly_probability:
            # Anomaly: large deviation
            value = self.baseline_mean + np.random.choice([-1, 1]) * self.anomaly_magnitude
        else:
            # Normal: sample from Gaussian
            value = np.random.normal(self.baseline_mean, self.baseline_std)

        self.time_step += 1
        return float(value)


class VisionSensor(SensorSimulator):
    """
    Generates simulated vision data (image tensors)

    Characteristics:
    - Tensor output (testing multi-modal salience)
    - Spatial patterns (testing similarity computation)
    - Moving objects (testing novelty detection)
    """

    def __init__(
        self,
        sensor_id: str,
        image_size: tuple = (32, 32),
        channels: int = 3,
        pattern_type: str = "moving_blob",  # "static", "moving_blob", "noise"
        sample_rate: float = 10.0
    ):
        super().__init__(sensor_id, sample_rate)
        self.image_size = image_size
        self.channels = channels
        self.pattern_type = pattern_type

        # For moving blob
        self.blob_position = np.array([image_size[0] // 2, image_size[1] // 2], dtype=float)
        self.blob_velocity = np.array([0.5, 0.3], dtype=float)

    def get_sample(self) -> torch.Tensor:
        """Generate image tensor"""
        if self.pattern_type == "static":
            # Static checkerboard pattern
            image = self._generate_checkerboard()

        elif self.pattern_type == "moving_blob":
            # Moving Gaussian blob
            image = self._generate_moving_blob()

        elif self.pattern_type == "noise":
            # Random noise
            image = torch.rand(self.channels, *self.image_size)

        else:
            raise ValueError(f"Unknown pattern_type: {self.pattern_type}")

        self.time_step += 1
        return image

    def _generate_checkerboard(self) -> torch.Tensor:
        """Generate checkerboard pattern"""
        image = torch.zeros(self.channels, *self.image_size)
        square_size = 4

        for i in range(0, self.image_size[0], square_size):
            for j in range(0, self.image_size[1], square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    image[:, i:i+square_size, j:j+square_size] = 1.0

        return image

    def _generate_moving_blob(self) -> torch.Tensor:
        """Generate moving Gaussian blob"""
        image = torch.zeros(self.channels, *self.image_size)

        # Update position
        self.blob_position += self.blob_velocity

        # Bounce off edges
        if self.blob_position[0] < 5 or self.blob_position[0] > self.image_size[0] - 5:
            self.blob_velocity[0] *= -1
        if self.blob_position[1] < 5 or self.blob_position[1] > self.image_size[1] - 5:
            self.blob_velocity[1] *= -1

        # Draw Gaussian blob
        y, x = np.ogrid[0:self.image_size[0], 0:self.image_size[1]]
        blob_center = self.blob_position.astype(int)

        dist_sq = (x - blob_center[1])**2 + (y - blob_center[0])**2
        blob = np.exp(-dist_sq / 20.0)

        for c in range(self.channels):
            image[c] = torch.from_numpy(blob).float()

        return image


class GoalRelevantSensor(SensorSimulator):
    """
    Generates signal correlated with goal achievement

    Characteristics:
    - High reward (predicts positive outcomes)
    - Learned through outcome feedback
    - Can test reward estimator learning
    """

    def __init__(
        self,
        sensor_id: str,
        goal_pattern: Callable[[int], float],
        noise_level: float = 0.2,
        sample_rate: float = 10.0
    ):
        """
        Args:
            goal_pattern: Function mapping time_step -> goal-relevant value
            noise_level: Amount of noise to add
        """
        super().__init__(sensor_id, sample_rate)
        self.goal_pattern = goal_pattern
        self.noise_level = noise_level

    def get_sample(self) -> float:
        """Generate goal-relevant value"""
        value = self.goal_pattern(self.time_step)

        if self.noise_level > 0:
            value += np.random.normal(0, self.noise_level)

        self.time_step += 1
        return float(value)


class MultiSensorEnvironment:
    """
    Manages multiple sensors and generates coordinated outputs

    Useful for testing cross-sensor conflict detection
    """

    def __init__(self, sensors: Dict[str, SensorSimulator]):
        self.sensors = sensors

    def get_snapshot(self) -> Dict[str, Any]:
        """Get current reading from all sensors"""
        return {
            sensor_id: sensor.get_sample()
            for sensor_id, sensor in self.sensors.items()
        }

    def reset_all(self):
        """Reset all sensors"""
        for sensor in self.sensors.values():
            sensor.reset()


# Predefined test scenarios

def create_calm_environment() -> MultiSensorEnvironment:
    """
    Low salience environment: predictable, familiar patterns

    Expected SNARC response:
    - Low surprise, novelty, arousal
    - Suggested stance: CONFIDENT_EXECUTION
    """
    return MultiSensorEnvironment({
        'temp_sensor': PeriodicSensor('temp_sensor', frequency=0.5, amplitude=2.0, noise_level=0.1),
        'pressure_sensor': PeriodicSensor('pressure_sensor', frequency=0.5, amplitude=3.0, noise_level=0.1),
        'light_sensor': PeriodicSensor('light_sensor', frequency=0.5, amplitude=1.5, noise_level=0.05)
    })


def create_surprising_environment() -> MultiSensorEnvironment:
    """
    High surprise environment: sudden changes

    Expected SNARC response:
    - High surprise, arousal
    - Suggested stance: CURIOUS_UNCERTAINTY or EXPLORATORY
    """
    return MultiSensorEnvironment({
        'accel_x': StepChangeSensor('accel_x', step_value=5.0, step_interval=30),
        'accel_y': StepChangeSensor('accel_y', step_value=3.0, step_interval=45),
        'gyro': PeriodicSensor('gyro', frequency=1.0, amplitude=2.0, noise_level=0.1)
    })


def create_novel_environment() -> MultiSensorEnvironment:
    """
    High novelty environment: never-seen-before patterns

    Expected SNARC response:
    - High novelty
    - Suggested stance: CURIOUS_UNCERTAINTY
    """
    return MultiSensorEnvironment({
        'sensor_a': RandomWalkSensor('sensor_a', step_size=2.0),  # Larger steps for more novelty
        'sensor_b': RandomWalkSensor('sensor_b', step_size=1.5),
        'sensor_c': AnomalySensor('sensor_c', anomaly_probability=0.15)
    })


def create_conflicting_environment() -> MultiSensorEnvironment:
    """
    High conflict environment: sensors disagree

    Expected SNARC response:
    - High conflict
    - Suggested stance: SKEPTICAL_VERIFICATION
    """
    return MultiSensorEnvironment({
        'vision': VisionSensor('vision', pattern_type='moving_blob'),
        'motion': PeriodicSensor('motion', frequency=2.0, amplitude=1.0),  # Different frequency
        'audio': StepChangeSensor('audio', step_interval=20)  # Uncorrelated steps
    })


def create_goal_relevant_environment() -> MultiSensorEnvironment:
    """
    High reward environment: goal-relevant signals

    Expected SNARC response:
    - High reward (after learning)
    - Suggested stance: FOCUSED_ATTENTION
    """
    # Goal: detect increasing trend
    goal_pattern = lambda t: 0.1 * t + np.sin(0.1 * t)

    return MultiSensorEnvironment({
        'target_metric': GoalRelevantSensor('target_metric', goal_pattern, noise_level=0.2),
        'distractor_1': PeriodicSensor('distractor_1', frequency=0.5),
        'distractor_2': RandomWalkSensor('distractor_2', step_size=0.3)
    })
