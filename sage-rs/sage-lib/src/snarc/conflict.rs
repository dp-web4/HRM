use std::collections::HashMap;

use crate::snarc::temporal::default_half_life;

const LN2: f64 = std::f64::consts::LN_2;

/// Conflict detector — measures cross-sensor disagreement by tracking
/// deviations from expected pairwise correlation patterns.
pub struct ConflictDetector {
    correlation_window: usize,
    half_life: f64,
    sensor_buffer: HashMap<String, Vec<f64>>,
    expected_correlations: HashMap<(String, String), f64>,
    last_update_time: HashMap<(String, String), f64>,
}

impl ConflictDetector {
    pub fn new(correlation_window: usize, half_life: Option<f64>) -> Self {
        Self {
            correlation_window,
            half_life: half_life.unwrap_or_else(|| default_half_life("conflict")),
            sensor_buffer: HashMap::new(),
            expected_correlations: HashMap::new(),
            last_update_time: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(20, None)
    }

    /// Compute conflict for `sensor_id` given all current sensor readings.
    /// Returns 0.0 (agreement) to 1.0 (high conflict).
    pub fn compute(
        &mut self,
        all_sensor_outputs: &HashMap<String, f64>,
        sensor_id: &str,
    ) -> f64 {
        let now = crate::snarc::temporal::now_secs();

        // Update buffers for all sensors
        for (sid, &value) in all_sensor_outputs {
            let buf = self.sensor_buffer.entry(sid.clone()).or_insert_with(Vec::new);
            buf.push(value);
            if buf.len() > self.correlation_window {
                let excess = buf.len() - self.correlation_window;
                buf.drain(..excess);
            }
        }

        if all_sensor_outputs.len() < 2 {
            return 0.0;
        }

        let mut conflicts = Vec::new();

        for (other_id, _) in all_sensor_outputs {
            if other_id == sensor_id {
                continue;
            }

            let buf_self = match self.sensor_buffer.get(sensor_id) {
                Some(b) if b.len() >= 5 => b,
                _ => continue,
            };
            let buf_other = match self.sensor_buffer.get(other_id.as_str()) {
                Some(b) if b.len() >= 5 => b,
                _ => continue,
            };

            let current_corr = pearson_correlation(buf_self, buf_other);
            let key = sorted_key(sensor_id, other_id);
            let expected_corr = self.expected_correlations.get(&key).copied().unwrap_or(0.0);

            let conflict = (current_corr - expected_corr).abs();
            conflicts.push(conflict);

            // Update expected correlation with time-adaptive EMA
            let last_t = self.last_update_time.get(&key).copied().unwrap_or(now);
            let elapsed = (now - last_t).max(0.0);
            let alpha = if self.half_life > 0.0 {
                (1.0 - (-LN2 * elapsed / self.half_life).exp()).clamp(0.01, 0.5)
            } else {
                0.1
            };

            let new_expected = match self.expected_correlations.get(&key) {
                Some(&old) => alpha * current_corr + (1.0 - alpha) * old,
                None => current_corr,
            };
            self.expected_correlations.insert(key.clone(), new_expected);
            self.last_update_time.insert(key, now);
        }

        if conflicts.is_empty() {
            return 0.0;
        }

        conflicts.iter().cloned().fold(0.0_f64, f64::max).clamp(0.0, 1.0)
    }

    pub fn reset_sensor(&mut self, sensor_id: &str) {
        self.sensor_buffer.remove(sensor_id);
        let keys_to_remove: Vec<_> = self.expected_correlations
            .keys()
            .filter(|(a, b)| a == sensor_id || b == sensor_id)
            .cloned()
            .collect();
        for key in keys_to_remove {
            self.expected_correlations.remove(&key);
            self.last_update_time.remove(&key);
        }
    }

    pub fn reset_all(&mut self) {
        self.sensor_buffer.clear();
        self.expected_correlations.clear();
        self.last_update_time.clear();
    }
}

fn sorted_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Pearson correlation between two sequences. Returns 0.0 on degenerate input.
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }

    let mean_a: f64 = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }

    (cov / denom).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_sensor_no_conflict() {
        let mut det = ConflictDetector::with_defaults();
        let mut readings = HashMap::new();
        readings.insert("only".to_string(), 5.0);
        assert_eq!(det.compute(&readings, "only"), 0.0);
    }

    #[test]
    fn correlated_sensors_low_conflict() {
        let mut det = ConflictDetector::with_defaults();
        // Build correlated history
        for i in 0..10 {
            let v = i as f64;
            let mut readings = HashMap::new();
            readings.insert("a".to_string(), v);
            readings.insert("b".to_string(), v * 2.0);
            det.compute(&readings, "a");
        }
        // Continue the pattern — should be low conflict
        let mut readings = HashMap::new();
        readings.insert("a".to_string(), 10.0);
        readings.insert("b".to_string(), 20.0);
        let c = det.compute(&readings, "a");
        assert!(c < 0.3, "correlated conflict={c}, expected low");
    }

    #[test]
    fn sudden_decorrelation_high_conflict() {
        let mut det = ConflictDetector::with_defaults();
        // Build correlated history
        for i in 0..10 {
            let v = i as f64;
            let mut readings = HashMap::new();
            readings.insert("a".to_string(), v);
            readings.insert("b".to_string(), v);
            det.compute(&readings, "a");
        }
        // Now invert the relationship
        for i in 10..20 {
            let v = i as f64;
            let mut readings = HashMap::new();
            readings.insert("a".to_string(), v);
            readings.insert("b".to_string(), -v);
            det.compute(&readings, "a");
        }
        // The correlation flipped — should detect conflict
        let mut readings = HashMap::new();
        readings.insert("a".to_string(), 20.0);
        readings.insert("b".to_string(), -20.0);
        let c = det.compute(&readings, "a");
        assert!(c > 0.3, "decorrelated conflict={c}, expected elevated");
    }

    #[test]
    fn pearson_perfect_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_correlation(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_negative_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson_correlation(&a, &b) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn pearson_constant_returns_zero() {
        let a = vec![5.0, 5.0, 5.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(pearson_correlation(&a, &b), 0.0);
    }
}
