use std::collections::HashMap;

use crate::snarc::temporal::{default_half_life, TimestampedDeque, now_secs};

/// Simple Exponential Moving Average predictor.
/// Predicts next value as weighted average of past observations.
pub struct SimplePredictorEMA {
    alpha: f64,
    prediction: Option<f64>,
}

impl SimplePredictorEMA {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            prediction: None,
        }
    }

    pub fn predict(&self) -> Option<f64> {
        self.prediction
    }

    pub fn update(&mut self, observation: f64) {
        self.prediction = Some(match self.prediction {
            None => observation,
            Some(prev) => self.alpha * observation + (1.0 - self.alpha) * prev,
        });
    }
}

/// Surprise detector — measures prediction error against an EMA baseline,
/// normalizes via time-weighted percentile rank over recent history.
pub struct SurpriseDetector {
    alpha: f64,
    history_size: usize,
    half_life: f64,
    predictors: HashMap<String, SimplePredictorEMA>,
    timed_history: HashMap<String, TimestampedDeque>,
}

impl SurpriseDetector {
    pub fn new(alpha: f64, history_size: usize, half_life: Option<f64>) -> Self {
        Self {
            alpha,
            history_size,
            half_life: half_life.unwrap_or_else(|| default_half_life("surprise")),
            predictors: HashMap::new(),
            timed_history: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.3, 100, None)
    }

    /// Compute surprise for a scalar observation on the given sensor.
    /// Returns a score in [0.0, 1.0].
    pub fn compute(&mut self, observation: f64, sensor_id: &str) -> f64 {
        if !self.predictors.contains_key(sensor_id) {
            self.predictors
                .insert(sensor_id.to_string(), SimplePredictorEMA::new(self.alpha));
            self.timed_history.insert(
                sensor_id.to_string(),
                TimestampedDeque::new(self.history_size),
            );
        }

        let prediction = self.predictors.get(sensor_id).unwrap().predict();

        if prediction.is_none() {
            self.predictors.get_mut(sensor_id).unwrap().update(observation);
            return 0.0;
        }

        let predicted = prediction.unwrap();
        let error = (observation - predicted).abs();
        let normalized = self.normalize_surprise(error, sensor_id);

        let now = now_secs();
        self.predictors.get_mut(sensor_id).unwrap().update(observation);
        self.timed_history
            .get_mut(sensor_id)
            .unwrap()
            .append(error, Some(now));

        normalized
    }

    fn normalize_surprise(&self, error: f64, sensor_id: &str) -> f64 {
        let timed = match self.timed_history.get(sensor_id) {
            Some(t) if t.len() >= 10 => t,
            _ => return (error / 10.0).min(1.0),
        };

        timed.weighted_percentile_of(error, self.half_life, None)
    }

    pub fn reset_sensor(&mut self, sensor_id: &str) {
        self.predictors.remove(sensor_id);
        self.timed_history.remove(sensor_id);
    }

    pub fn reset_all(&mut self) {
        self.predictors.clear();
        self.timed_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_first_observation() {
        let mut ema = SimplePredictorEMA::new(0.3);
        assert!(ema.predict().is_none());
        ema.update(5.0);
        assert_eq!(ema.predict(), Some(5.0));
    }

    #[test]
    fn ema_converges_toward_constant() {
        let mut ema = SimplePredictorEMA::new(0.3);
        for _ in 0..100 {
            ema.update(10.0);
        }
        assert!((ema.predict().unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn ema_tracks_step_change() {
        let mut ema = SimplePredictorEMA::new(0.3);
        for _ in 0..50 {
            ema.update(1.0);
        }
        // Step to 10.0
        for _ in 0..50 {
            ema.update(10.0);
        }
        // Should be very close to 10.0 after 50 updates at alpha=0.3
        assert!((ema.predict().unwrap() - 10.0).abs() < 0.01);
    }

    #[test]
    fn surprise_first_observation_is_zero() {
        let mut det = SurpriseDetector::with_defaults();
        assert_eq!(det.compute(5.0, "test"), 0.0);
    }

    #[test]
    fn surprise_constant_stream_low() {
        let mut det = SurpriseDetector::with_defaults();
        // Feed 20 identical observations to build history
        for _ in 0..20 {
            det.compute(5.0, "test");
        }
        // Next identical observation should be low surprise
        let s = det.compute(5.0, "test");
        assert!(s < 0.1, "constant stream surprise={s}, expected <0.1");
    }

    #[test]
    fn surprise_spike_is_high() {
        let mut det = SurpriseDetector::with_defaults();
        // Build baseline
        for _ in 0..30 {
            det.compute(1.0, "test");
        }
        // Big spike
        let s = det.compute(100.0, "test");
        assert!(s > 0.8, "spike surprise={s}, expected >0.8");
    }

    #[test]
    fn surprise_independent_sensors() {
        let mut det = SurpriseDetector::with_defaults();
        // First obs on each sensor → 0.0
        assert_eq!(det.compute(5.0, "a"), 0.0);
        assert_eq!(det.compute(100.0, "b"), 0.0);

        // Sensor "a" gets constant, "b" gets spike relative to itself
        for _ in 0..20 {
            det.compute(5.0, "a");
            det.compute(100.0, "b");
        }
        // Both should have low surprise for their respective constants
        let sa = det.compute(5.0, "a");
        let sb = det.compute(100.0, "b");
        assert!(sa < 0.1, "sensor a surprise={sa}");
        assert!(sb < 0.1, "sensor b surprise={sb}");
    }

    #[test]
    fn surprise_reset_sensor() {
        let mut det = SurpriseDetector::with_defaults();
        for _ in 0..20 {
            det.compute(5.0, "test");
        }
        det.reset_sensor("test");
        // After reset, first observation is again zero
        assert_eq!(det.compute(5.0, "test"), 0.0);
    }

    #[test]
    fn surprise_20_observation_parity() {
        // Feed a sequence of 20 scalar observations and verify the detector
        // produces reasonable monotonic surprise for divergent values.
        let mut det = SurpriseDetector::new(0.3, 100, Some(60.0));
        let observations = [
            1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.02, 0.98, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let mut scores = Vec::new();
        for &obs in &observations {
            scores.push(det.compute(obs, "parity"));
        }

        // After 20 near-constant observations, a big deviation should spike
        let spike = det.compute(5.0, "parity");
        let last_calm = *scores.last().unwrap();
        assert!(
            spike > last_calm,
            "spike ({spike}) should exceed calm ({last_calm})"
        );
    }
}
