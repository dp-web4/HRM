use std::time::{SystemTime, UNIX_EPOCH};

const LN2: f64 = std::f64::consts::LN_2;

pub const DEFAULT_HALF_LIVES: &[(&str, f64)] = &[
    ("surprise", 60.0),
    ("novelty", 300.0),
    ("arousal", 120.0),
    ("reward", 600.0),
    ("conflict", 300.0),
];

pub fn default_half_life(detector: &str) -> f64 {
    DEFAULT_HALF_LIVES
        .iter()
        .find(|(name, _)| *name == detector)
        .map(|(_, hl)| *hl)
        .unwrap_or(120.0)
}

/// Exponential decay weight for an observation.
/// Returns 1.0 when obs_time == now, 0.5 at one half-life, approaches 0.
pub fn time_decay_weight(obs_time: f64, now: f64, half_life: f64) -> f64 {
    if half_life <= 0.0 {
        return 1.0;
    }
    let elapsed = (now - obs_time).max(0.0);
    (-LN2 * elapsed / half_life).exp()
}

/// Time-weighted percentile rank of `target` within `values`.
/// Returns fraction of weighted observations that fall below target, in [0, 1].
pub fn weighted_percentile(values: &[f64], weights: &[f64], target: f64) -> f64 {
    if values.is_empty() || weights.is_empty() {
        return 0.5;
    }

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return 0.5;
    }

    let below_weight: f64 = values
        .iter()
        .zip(weights.iter())
        .filter(|(v, _)| **v < target)
        .map(|(_, w)| *w)
        .sum();

    below_weight / total_weight
}

pub fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Fixed-capacity buffer storing (value, timestamp) pairs with time-decay weighting.
pub struct TimestampedDeque {
    maxlen: usize,
    values: Vec<f64>,
    timestamps: Vec<f64>,
}

impl TimestampedDeque {
    pub fn new(maxlen: usize) -> Self {
        Self {
            maxlen,
            values: Vec::with_capacity(maxlen),
            timestamps: Vec::with_capacity(maxlen),
        }
    }

    pub fn append(&mut self, value: f64, timestamp: Option<f64>) {
        let ts = timestamp.unwrap_or_else(now_secs);
        self.values.push(value);
        self.timestamps.push(ts);
        if self.values.len() > self.maxlen {
            self.values.remove(0);
            self.timestamps.remove(0);
        }
    }

    /// Returns (value, weight) pairs with time-decay weights.
    pub fn weighted_values(&self, half_life: f64, now: Option<f64>) -> Vec<(f64, f64)> {
        let now = now.unwrap_or_else(now_secs);
        self.values
            .iter()
            .zip(self.timestamps.iter())
            .map(|(v, t)| (*v, time_decay_weight(*t, now, half_life)))
            .collect()
    }

    /// Percentile rank of `target` in this buffer, weighted by recency.
    pub fn weighted_percentile_of(
        &self,
        target: f64,
        half_life: f64,
        now: Option<f64>,
    ) -> f64 {
        let now = now.unwrap_or_else(now_secs);
        let weights: Vec<f64> = self
            .timestamps
            .iter()
            .map(|t| time_decay_weight(*t, now, half_life))
            .collect();
        weighted_percentile(&self.values, &weights, target)
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn timestamps(&self) -> &[f64] {
        &self.timestamps
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.timestamps.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_at_zero_elapsed() {
        assert!((time_decay_weight(100.0, 100.0, 60.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn decay_at_one_half_life() {
        assert!((time_decay_weight(0.0, 60.0, 60.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn decay_at_two_half_lives() {
        assert!((time_decay_weight(0.0, 120.0, 60.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn decay_zero_half_life_returns_one() {
        assert_eq!(time_decay_weight(0.0, 1000.0, 0.0), 1.0);
    }

    #[test]
    fn decay_negative_half_life_returns_one() {
        assert_eq!(time_decay_weight(0.0, 1000.0, -5.0), 1.0);
    }

    #[test]
    fn decay_future_obs_clamped() {
        // obs_time > now → elapsed clamped to 0 → weight 1.0
        assert!((time_decay_weight(200.0, 100.0, 60.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_empty() {
        assert_eq!(weighted_percentile(&[], &[], 5.0), 0.5);
    }

    #[test]
    fn percentile_uniform_weights() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0; 5];

        // target=3.0 → 2 values below (1,2) → 2/5 = 0.4
        assert!((weighted_percentile(&values, &weights, 3.0) - 0.4).abs() < 1e-12);

        // target=6.0 → all 5 below → 5/5 = 1.0
        assert!((weighted_percentile(&values, &weights, 6.0) - 1.0).abs() < 1e-12);

        // target=0.0 → none below → 0.0
        assert!((weighted_percentile(&values, &weights, 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_nonuniform_weights() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![0.5, 0.3, 0.2]; // total = 1.0

        // target=2.5 → values below: 1.0 (w=0.5), 2.0 (w=0.3) → 0.8/1.0 = 0.8
        assert!((weighted_percentile(&values, &weights, 2.5) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn deque_append_and_evict() {
        let mut d = TimestampedDeque::new(3);
        d.append(1.0, Some(100.0));
        d.append(2.0, Some(200.0));
        d.append(3.0, Some(300.0));
        assert_eq!(d.len(), 3);

        d.append(4.0, Some(400.0));
        assert_eq!(d.len(), 3);
        assert_eq!(d.values(), &[2.0, 3.0, 4.0]);
        assert_eq!(d.timestamps(), &[200.0, 300.0, 400.0]);
    }

    #[test]
    fn deque_weighted_percentile_matches_python() {
        // Reproduce a fixed scenario: 5 observations at known times,
        // query percentile at a known "now" with surprise half-life (60s).
        let mut d = TimestampedDeque::new(100);
        d.append(0.1, Some(1000.0));
        d.append(0.5, Some(1020.0));
        d.append(0.3, Some(1040.0));
        d.append(0.8, Some(1050.0));
        d.append(0.2, Some(1055.0));

        let now = 1060.0;
        let half_life = 60.0;

        // Compute weights manually:
        // w0 = exp(-ln2 * 60/60) = 0.5
        // w1 = exp(-ln2 * 40/60) = exp(-0.4621) ≈ 0.6300
        // w2 = exp(-ln2 * 20/60) = exp(-0.2310) ≈ 0.7937
        // w3 = exp(-ln2 * 10/60) = exp(-0.1155) ≈ 0.8909
        // w4 = exp(-ln2 *  5/60) = exp(-0.0578) ≈ 0.9439
        // total ≈ 3.7585

        // target=0.4 → values below: 0.1 (w0=0.5), 0.3 (w2=0.7937), 0.2 (w4=0.9439)
        // below_weight ≈ 2.2376
        // percentile ≈ 2.2376/3.7585 ≈ 0.5954

        let p = d.weighted_percentile_of(0.4, half_life, Some(now));
        assert!(
            (p - 0.5954).abs() < 0.001,
            "expected ~0.5954, got {p}"
        );
    }

    #[test]
    fn deque_clear() {
        let mut d = TimestampedDeque::new(10);
        d.append(1.0, Some(0.0));
        d.append(2.0, Some(1.0));
        assert_eq!(d.len(), 2);
        d.clear();
        assert!(d.is_empty());
    }
}
