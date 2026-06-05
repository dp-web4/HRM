use std::collections::HashMap;

use crate::snarc::temporal::{default_half_life, time_decay_weight, now_secs};

struct OutcomeEntry {
    observation: f64,
    reward: f64,
    timestamp: f64,
}

/// Reward estimator — learns associations between observations and outcomes,
/// predicts goal relevance via similarity-weighted retrieval.
pub struct RewardEstimator {
    memory_size: usize,
    half_life: f64,
    outcome_memory: HashMap<String, Vec<OutcomeEntry>>,
}

impl RewardEstimator {
    pub fn new(memory_size: usize, half_life: Option<f64>) -> Self {
        Self {
            memory_size,
            half_life: half_life.unwrap_or_else(|| default_half_life("reward")),
            outcome_memory: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(500, None)
    }

    /// Estimate reward relevance. Returns 0.0 (irrelevant) to 1.0 (highly relevant).
    /// Returns 0.5 when there's no outcome history.
    pub fn compute(&self, observation: f64, sensor_id: &str) -> f64 {
        let entries = match self.outcome_memory.get(sensor_id) {
            Some(e) if !e.is_empty() => e,
            _ => return 0.5,
        };

        let similar = self.find_similar_outcomes(observation, entries, 10);
        if similar.is_empty() {
            return 0.5;
        }

        let avg_reward: f64 = similar.iter().map(|&r| r).sum::<f64>() / similar.len() as f64;
        // Normalize from [-1, 1] to [0, 1]
        ((avg_reward + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Record an outcome: the observation that was made, and the reward received.
    pub fn update_from_outcome(&mut self, observation: f64, sensor_id: &str, reward: f64) {
        let entries = self.outcome_memory
            .entry(sensor_id.to_string())
            .or_insert_with(Vec::new);

        entries.push(OutcomeEntry {
            observation,
            reward,
            timestamp: now_secs(),
        });

        if entries.len() > self.memory_size {
            let excess = entries.len() - self.memory_size;
            entries.drain(..excess);
        }
    }

    fn find_similar_outcomes(&self, observation: f64, entries: &[OutcomeEntry], top_k: usize) -> Vec<f64> {
        let now = now_secs();

        let mut scored: Vec<(f64, f64)> = entries
            .iter()
            .map(|e| {
                let sim = scalar_similarity(observation, e.observation);
                let recency = time_decay_weight(e.timestamp, now, self.half_life);
                (sim * recency, e.reward)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(top_k).map(|&(_, r)| r).collect()
    }

    pub fn reset_sensor(&mut self, sensor_id: &str) {
        self.outcome_memory.remove(sensor_id);
    }

    pub fn reset_all(&mut self) {
        self.outcome_memory.clear();
    }
}

fn scalar_similarity(a: f64, b: f64) -> f64 {
    (-(a - b).abs() / 10.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_history_returns_neutral() {
        let est = RewardEstimator::with_defaults();
        assert_eq!(est.compute(5.0, "test"), 0.5);
    }

    #[test]
    fn positive_outcomes_predict_high() {
        let mut est = RewardEstimator::with_defaults();
        for _ in 0..20 {
            est.update_from_outcome(5.0, "test", 1.0);
        }
        let r = est.compute(5.0, "test");
        assert!(r > 0.9, "positive reward={r}, expected high");
    }

    #[test]
    fn negative_outcomes_predict_low() {
        let mut est = RewardEstimator::with_defaults();
        for _ in 0..20 {
            est.update_from_outcome(5.0, "test", -1.0);
        }
        let r = est.compute(5.0, "test");
        assert!(r < 0.1, "negative reward={r}, expected low");
    }

    #[test]
    fn mixed_outcomes_predict_by_similarity() {
        let mut est = RewardEstimator::with_defaults();
        // Observations near 5.0 → positive reward
        for _ in 0..10 {
            est.update_from_outcome(5.0, "test", 1.0);
        }
        // Observations near 50.0 → negative reward
        for _ in 0..10 {
            est.update_from_outcome(50.0, "test", -1.0);
        }
        // Query near 5.0 → should predict high reward
        let r_near_pos = est.compute(6.0, "test");
        // Query near 50.0 → should predict low reward
        let r_near_neg = est.compute(49.0, "test");
        assert!(r_near_pos > r_near_neg, "pos={r_near_pos}, neg={r_near_neg}");
    }

    #[test]
    fn memory_eviction() {
        let mut est = RewardEstimator::new(5, Some(600.0));
        for i in 0..10 {
            est.update_from_outcome(i as f64, "test", 0.5);
        }
        assert_eq!(est.outcome_memory.get("test").unwrap().len(), 5);
    }
}
