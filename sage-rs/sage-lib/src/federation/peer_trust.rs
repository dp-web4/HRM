use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T3Scores {
    pub talent: f64,
    pub training: f64,
    pub temperament: f64,
    #[serde(default)]
    pub interactions: u64,
    #[serde(default)]
    pub last_updated: f64,
}

impl Default for T3Scores {
    fn default() -> Self {
        Self {
            talent: 0.5,
            training: 0.5,
            temperament: 0.5,
            interactions: 0,
            last_updated: 0.0,
        }
    }
}

impl T3Scores {
    /// Geometric mean of T3 dimensions — single reputation scalar.
    pub fn reputation(&self) -> f64 {
        (self.talent * self.training * self.temperament).cbrt()
    }
}

struct OutcomeDeltas {
    talent: f64,
    training: f64,
    temperament: f64,
}

fn outcome_deltas(outcome: &str) -> Option<OutcomeDeltas> {
    match outcome {
        "success"      => Some(OutcomeDeltas { talent: 0.05, training: 0.02, temperament: 0.05 }),
        "timeout"      => Some(OutcomeDeltas { talent: 0.0, training: 0.0, temperament: -0.10 }),
        "error"        => Some(OutcomeDeltas { talent: -0.05, training: -0.02, temperament: -0.05 }),
        "quality_high" => Some(OutcomeDeltas { talent: 0.10, training: 0.05, temperament: 0.02 }),
        "quality_low"  => Some(OutcomeDeltas { talent: -0.08, training: -0.03, temperament: -0.02 }),
        _ => None,
    }
}

fn now_epoch() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[derive(Serialize, Deserialize)]
struct PeerTrustState {
    peers: HashMap<String, T3Scores>,
    #[serde(default)]
    last_updated: f64,
}

/// Per-peer T3 trust tracker with JSON persistence.
pub struct PeerTrustTracker {
    state_path: PathBuf,
    alpha: f64,
    peers: HashMap<String, T3Scores>,
}

impl PeerTrustTracker {
    pub fn new(state_path: &Path, alpha: f64) -> Self {
        let mut tracker = Self {
            state_path: state_path.to_path_buf(),
            alpha,
            peers: HashMap::new(),
        };
        tracker.load();
        tracker
    }

    pub fn with_defaults(state_path: &Path) -> Self {
        Self::new(state_path, 0.1)
    }

    fn ensure_peer(&mut self, name: &str) {
        if !self.peers.contains_key(name) {
            self.peers.insert(name.to_string(), T3Scores {
                last_updated: now_epoch(),
                ..Default::default()
            });
        }
    }

    /// Record an interaction outcome and update T3 scores via EMA.
    pub fn record_interaction(&mut self, machine_name: &str, outcome: &str) {
        let deltas = match outcome_deltas(outcome) {
            Some(d) => d,
            None => return,
        };

        self.ensure_peer(machine_name);
        let peer = self.peers.get_mut(machine_name).unwrap();

        peer.talent = (peer.talent + self.alpha * deltas.talent).clamp(0.0, 1.0);
        peer.training = (peer.training + self.alpha * deltas.training).clamp(0.0, 1.0);
        peer.temperament = (peer.temperament + self.alpha * deltas.temperament).clamp(0.0, 1.0);
        peer.interactions += 1;
        peer.last_updated = now_epoch();

        if peer.interactions % 5 == 0 {
            self.save();
        }
    }

    pub fn get_trust(&mut self, machine_name: &str) -> &T3Scores {
        self.ensure_peer(machine_name);
        self.peers.get(machine_name).unwrap()
    }

    pub fn get_reputation(&mut self, machine_name: &str) -> f64 {
        self.get_trust(machine_name).reputation()
    }

    pub fn get_all_trust(&self) -> &HashMap<String, T3Scores> {
        &self.peers
    }

    pub fn save(&self) {
        let state = PeerTrustState {
            peers: self.peers.clone(),
            last_updated: now_epoch(),
        };
        if let Some(parent) = self.state_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let json = serde_json::to_string_pretty(&state).unwrap_or_default();
        let _ = std::fs::write(&self.state_path, json);
    }

    fn load(&mut self) {
        let data = match std::fs::read_to_string(&self.state_path) {
            Ok(s) => s,
            Err(_) => return,
        };
        let state: PeerTrustState = match serde_json::from_str(&data) {
            Ok(s) => s,
            Err(_) => return,
        };
        self.peers = state.peers;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_path() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("sage-trust-test-{}-{n}.json", std::process::id()))
    }

    #[test]
    fn default_trust_is_neutral() {
        let path = temp_path();
        let mut tracker = PeerTrustTracker::with_defaults(&path);
        let t = tracker.get_trust("thor");
        assert_eq!(t.talent, 0.5);
        assert_eq!(t.training, 0.5);
        assert_eq!(t.temperament, 0.5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn success_increases_trust() {
        let path = temp_path();
        let mut tracker = PeerTrustTracker::with_defaults(&path);
        tracker.record_interaction("thor", "success");
        let t = tracker.get_trust("thor");
        assert!(t.talent > 0.5);
        assert!(t.temperament > 0.5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn error_decreases_trust() {
        let path = temp_path();
        let mut tracker = PeerTrustTracker::with_defaults(&path);
        tracker.record_interaction("thor", "error");
        let t = tracker.get_trust("thor");
        assert!(t.talent < 0.5);
        assert!(t.temperament < 0.5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn clamped_to_0_1() {
        let path = temp_path();
        let mut tracker = PeerTrustTracker::with_defaults(&path);
        for _ in 0..200 {
            tracker.record_interaction("thor", "quality_high");
        }
        let t = tracker.get_trust("thor");
        assert!(t.talent <= 1.0);
        assert!(t.training <= 1.0);
        assert!(t.temperament <= 1.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reputation_geometric_mean() {
        let scores = T3Scores {
            talent: 0.8,
            training: 0.6,
            temperament: 0.9,
            interactions: 0,
            last_updated: 0.0,
        };
        let rep = scores.reputation();
        let expected = (0.8_f64 * 0.6 * 0.9).cbrt();
        assert!((rep - expected).abs() < 1e-10);
    }

    #[test]
    fn persistence_roundtrip() {
        let path = temp_path();

        {
            let mut tracker = PeerTrustTracker::with_defaults(&path);
            for _ in 0..10 {
                tracker.record_interaction("thor", "success");
            }
            tracker.save();
        }

        {
            let mut tracker = PeerTrustTracker::with_defaults(&path);
            let t = tracker.get_trust("thor");
            assert!(t.talent > 0.5);
            assert_eq!(t.interactions, 10);
        }

        let _ = std::fs::remove_file(&path);
    }
}
