use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorObservation {
    pub sensor_id: String,
    pub modality: String,
    pub data: serde_json::Value,
    pub timestamp: f64,
    pub trust: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SalienceScore {
    pub surprise: f64,
    pub novelty: f64,
    pub arousal: f64,
    pub reward: f64,
    pub conflict: f64,
    pub total: f64,
}

impl SalienceScore {
    pub fn from_components(surprise: f64, novelty: f64, arousal: f64, reward: f64, conflict: f64) -> Self {
        let total = (surprise + novelty + arousal + reward + conflict) / 5.0;
        Self { surprise, novelty, arousal, reward, conflict, total }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustPosture {
    pub confidence: f64,
    pub asymmetry: f64,
    pub breadth: f64,
    pub dominant_modality: String,
}

impl Default for TrustPosture {
    fn default() -> Self {
        Self {
            confidence: 0.5,
            asymmetry: 0.0,
            breadth: 1.0,
            dominant_modality: "message".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salience_total_is_mean() {
        let s = SalienceScore::from_components(0.8, 0.6, 0.4, 0.2, 0.0);
        assert!((s.total - 0.4).abs() < 1e-10);
    }

    #[test]
    fn default_salience_is_zero() {
        let s = SalienceScore::default();
        assert_eq!(s.total, 0.0);
    }
}
