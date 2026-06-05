use std::collections::HashMap;

use crate::snarc::temporal::{default_half_life, time_decay_weight, now_secs};

/// Novelty detector — measures dissimilarity from past observations.
/// Maintains per-sensor episodic memory with time-decay weighting on similarity.
pub struct NoveltyDetector {
    memory_size: usize,
    comparison_samples: usize,
    half_life: f64,
    memory: HashMap<String, Vec<f64>>,
    timestamps: HashMap<String, Vec<f64>>,
}

impl NoveltyDetector {
    pub fn new(memory_size: usize, comparison_samples: usize, half_life: Option<f64>) -> Self {
        Self {
            memory_size,
            comparison_samples,
            half_life: half_life.unwrap_or_else(|| default_half_life("novelty")),
            memory: HashMap::new(),
            timestamps: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(1000, 50, None)
    }

    /// Compute novelty for a scalar observation. Returns 0.0 (familiar) to 1.0 (novel).
    pub fn compute(&mut self, observation: f64, sensor_id: &str) -> f64 {
        let now = now_secs();

        if !self.memory.contains_key(sensor_id) {
            self.memory.insert(sensor_id.to_string(), Vec::new());
            self.timestamps.insert(sensor_id.to_string(), Vec::new());
        }

        let mem = self.memory.get(sensor_id).unwrap();

        if mem.is_empty() {
            self.memory.get_mut(sensor_id).unwrap().push(observation);
            self.timestamps.get_mut(sensor_id).unwrap().push(now);
            return 1.0;
        }

        let n = mem.len();
        let start = if n > self.comparison_samples { n - self.comparison_samples } else { 0 };
        let ts = self.timestamps.get(sensor_id).unwrap();

        let mut max_effective_sim: f64 = 0.0;
        for i in start..n {
            let raw_sim = scalar_similarity(observation, mem[i]);
            let recency = time_decay_weight(ts[i], now, self.half_life);
            let effective = raw_sim * recency;
            if effective > max_effective_sim {
                max_effective_sim = effective;
            }
        }

        let novelty = (1.0 - max_effective_sim).clamp(0.0, 1.0);

        // Store observation
        let mem = self.memory.get_mut(sensor_id).unwrap();
        let ts = self.timestamps.get_mut(sensor_id).unwrap();
        mem.push(observation);
        ts.push(now);
        if mem.len() > self.memory_size {
            mem.remove(0);
            ts.remove(0);
        }

        novelty
    }

    pub fn memory_size_for(&self, sensor_id: &str) -> usize {
        self.memory.get(sensor_id).map_or(0, |m| m.len())
    }

    pub fn reset_sensor(&mut self, sensor_id: &str) {
        self.memory.remove(sensor_id);
        self.timestamps.remove(sensor_id);
    }

    pub fn reset_all(&mut self) {
        self.memory.clear();
        self.timestamps.clear();
    }
}

/// Scalar similarity: exp(-|a - b| / 10.0), matching Python's NoveltyDetector.
fn scalar_similarity(a: f64, b: f64) -> f64 {
    (-(a - b).abs() / 10.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_observation_is_novel() {
        let mut det = NoveltyDetector::with_defaults();
        assert_eq!(det.compute(5.0, "test"), 1.0);
    }

    #[test]
    fn identical_repeat_is_familiar() {
        let mut det = NoveltyDetector::with_defaults();
        det.compute(5.0, "test");
        let n = det.compute(5.0, "test");
        // similarity = exp(0) = 1.0, novelty = 0.0
        assert!(n < 0.01, "identical repeat novelty={n}, expected ~0.0");
    }

    #[test]
    fn distant_value_is_novel() {
        let mut det = NoveltyDetector::with_defaults();
        det.compute(0.0, "test");
        let n = det.compute(100.0, "test");
        // similarity = exp(-100/10) ≈ 0.0000454, novelty ≈ 1.0
        assert!(n > 0.99, "distant novelty={n}, expected ~1.0");
    }

    #[test]
    fn sensor_independence() {
        let mut det = NoveltyDetector::with_defaults();
        det.compute(5.0, "a");
        det.compute(100.0, "b");
        // Second obs on "a" same value → familiar
        let na = det.compute(5.0, "a");
        // Second obs on "b" same value → familiar
        let nb = det.compute(100.0, "b");
        assert!(na < 0.01);
        assert!(nb < 0.01);
    }

    #[test]
    fn memory_eviction() {
        let mut det = NoveltyDetector::new(5, 5, Some(300.0));
        for i in 0..10 {
            det.compute(i as f64, "test");
        }
        assert_eq!(det.memory_size_for("test"), 5);
    }

    #[test]
    fn similarity_function() {
        assert!((scalar_similarity(5.0, 5.0) - 1.0).abs() < 1e-12);
        assert!((scalar_similarity(0.0, 10.0) - (-1.0_f64).exp()).abs() < 1e-12);
    }
}
