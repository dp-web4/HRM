use std::collections::HashMap;

use crate::snarc::temporal::{default_half_life, TimestampedDeque, now_secs};

/// Arousal detector — measures signal magnitude normalized against recent history.
pub struct ArousalDetector {
    history_size: usize,
    half_life: f64,
    timed_history: HashMap<String, TimestampedDeque>,
}

impl ArousalDetector {
    pub fn new(history_size: usize, half_life: Option<f64>) -> Self {
        Self {
            history_size,
            half_life: half_life.unwrap_or_else(|| default_half_life("arousal")),
            timed_history: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(100, None)
    }

    /// Compute arousal for a scalar observation. Returns 0.0 (calm) to 1.0 (urgent).
    pub fn compute(&mut self, observation: f64, sensor_id: &str) -> f64 {
        if !self.timed_history.contains_key(sensor_id) {
            self.timed_history.insert(
                sensor_id.to_string(),
                TimestampedDeque::new(self.history_size),
            );
        }

        let magnitude = observation.abs();
        let normalized = self.normalize_arousal(magnitude, sensor_id);

        let now = now_secs();
        self.timed_history
            .get_mut(sensor_id)
            .unwrap()
            .append(magnitude, Some(now));

        normalized
    }

    fn normalize_arousal(&self, magnitude: f64, sensor_id: &str) -> f64 {
        let timed = match self.timed_history.get(sensor_id) {
            Some(t) if t.len() >= 10 => t,
            _ => return (magnitude / 10.0).min(1.0),
        };

        timed.weighted_percentile_of(magnitude, self.half_life, None)
    }

    pub fn reset_sensor(&mut self, sensor_id: &str) {
        self.timed_history.remove(sensor_id);
    }

    pub fn reset_all(&mut self) {
        self.timed_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_magnitude_low_arousal() {
        let mut det = ArousalDetector::with_defaults();
        // Build baseline with values around 1.0
        for _ in 0..20 {
            det.compute(1.0, "test");
        }
        let a = det.compute(1.0, "test");
        assert!(a < 0.2, "constant arousal={a}, expected low");
    }

    #[test]
    fn high_magnitude_high_arousal() {
        let mut det = ArousalDetector::with_defaults();
        for _ in 0..20 {
            det.compute(1.0, "test");
        }
        let a = det.compute(100.0, "test");
        assert!(a > 0.9, "spike arousal={a}, expected high");
    }

    #[test]
    fn negative_values_use_abs() {
        let mut det = ArousalDetector::with_defaults();
        for _ in 0..20 {
            det.compute(1.0, "test");
        }
        // -100.0 has same magnitude as 100.0
        let a = det.compute(-100.0, "test");
        assert!(a > 0.9, "negative spike arousal={a}, expected high");
    }

    #[test]
    fn sensor_independence() {
        let mut det = ArousalDetector::with_defaults();
        for _ in 0..15 {
            det.compute(1.0, "calm");
            det.compute(50.0, "intense");
        }
        let a_calm = det.compute(1.0, "calm");
        let a_intense = det.compute(50.0, "intense");
        assert!(a_calm < 0.2);
        assert!(a_intense < 0.2); // 50.0 is baseline for "intense"
    }
}
