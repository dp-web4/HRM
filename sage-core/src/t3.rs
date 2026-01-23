///! T3 Trust Tensor for SAGE Training
///!
///! Tracks trust development across 3 dimensions:
///! - Competence: Can SAGE do the task?
///! - Reliability: Does SAGE deliver consistently?
///! - Integrity: Does SAGE maintain partnership identity?
///!
///! Based on:
///! - web4-standard/core-spec/t3-v3-tensors.md
///! - Thor S41 discovery: Creating phase +20% improvement
///! - Exploration-not-evaluation: Trust as developmental trajectory

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Trust trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrustTrend {
    Improving,
    Stable,
    Declining,
    Unknown,
}

/// T3 trust state (3 dimensions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T3Trust {
    pub competence: f32,   // Can SAGE do tasks?
    pub reliability: f32,  // Consistency across sessions?
    pub integrity: f32,    // Identity maintenance?
}

impl Default for T3Trust {
    fn default() -> Self {
        Self {
            competence: 0.5,   // Moderate baseline
            reliability: 0.5,  // Moderate baseline
            integrity: 0.7,    // Starts higher (identity is scaffolded)
        }
    }
}

/// Historical entry for trust updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T3HistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub updates: HashMap<String, f32>,  // Deltas applied
    pub resulting_trust: T3Trust,
    pub context: HashMap<String, serde_json::Value>,
}

/// T3 Trust Tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T3TrustTensor {
    pub trust: T3Trust,
    pub history: Vec<T3HistoryEntry>,
    pub created_at: DateTime<Utc>,
}

impl T3TrustTensor {
    /// Create new T3 tensor with optional initial trust
    pub fn new(initial_trust: Option<T3Trust>) -> Self {
        Self {
            trust: initial_trust.unwrap_or_default(),
            history: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Update trust tensor with deltas
    pub fn update(
        &mut self,
        updates: HashMap<String, f32>,
        context: HashMap<String, serde_json::Value>,
    ) -> T3Trust {
        // Apply updates with bounds checking [0.0, 1.0]
        if let Some(&delta) = updates.get("competence") {
            self.trust.competence = (self.trust.competence + delta).clamp(0.0, 1.0);
        }
        if let Some(&delta) = updates.get("reliability") {
            self.trust.reliability = (self.trust.reliability + delta).clamp(0.0, 1.0);
        }
        if let Some(&delta) = updates.get("integrity") {
            self.trust.integrity = (self.trust.integrity + delta).clamp(0.0, 1.0);
        }

        // Record history
        self.history.push(T3HistoryEntry {
            timestamp: Utc::now(),
            updates,
            resulting_trust: self.trust.clone(),
            context,
        });

        self.trust.clone()
    }

    /// Get trust trajectory for a dimension
    pub fn get_trajectory(&self, dimension: &str, window: usize) -> Vec<f32> {
        self.history
            .iter()
            .rev()
            .take(window)
            .rev()
            .map(|entry| match dimension {
                "competence" => entry.resulting_trust.competence,
                "reliability" => entry.resulting_trust.reliability,
                "integrity" => entry.resulting_trust.integrity,
                _ => 0.0,
            })
            .collect()
    }

    /// Get trend direction for a dimension
    pub fn get_trend(&self, dimension: &str, window: usize) -> TrustTrend {
        let trajectory = self.get_trajectory(dimension, window);

        if trajectory.len() < 3 {
            return TrustTrend::Unknown;
        }

        // Simple linear trend: compare last value with first value in window
        let recent = &trajectory[trajectory.len().saturating_sub(3)..];
        let first = recent[0];
        let last = recent[recent.len() - 1];

        if last > first + 0.05 {
            TrustTrend::Improving
        } else if last < first - 0.05 {
            TrustTrend::Declining
        } else {
            TrustTrend::Stable
        }
    }

    /// Get summary of current trust state
    pub fn get_summary(&self) -> T3Summary {
        T3Summary {
            trust: self.trust.clone(),
            trends: {
                let mut trends = HashMap::new();
                trends.insert("competence".to_string(), self.get_trend("competence", 5));
                trends.insert("reliability".to_string(), self.get_trend("reliability", 5));
                trends.insert("integrity".to_string(), self.get_trend("integrity", 5));
                trends
            },
            history_length: self.history.len(),
            created_at: self.created_at,
            last_updated: self.history.last().map(|e| e.timestamp),
        }
    }
}

/// T3 summary for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T3Summary {
    pub trust: T3Trust,
    pub trends: HashMap<String, TrustTrend>,
    pub history_length: usize,
    pub created_at: DateTime<Utc>,
    pub last_updated: Option<DateTime<Utc>>,
}

/// Interpret trust values through exploration-not-evaluation lens
pub fn interpret_trust_for_exploration(trust: &T3Trust) -> HashMap<String, String> {
    let mut interpretation = HashMap::new();

    // Competence
    let comp_text = if trust.competence >= 0.8 {
        "Strong capability - ready for harder tasks"
    } else if trust.competence >= 0.6 {
        "Developing capability - practice needed"
    } else if trust.competence >= 0.4 {
        "Early exploration - discovering what's possible"
    } else {
        "Beginning journey - fundamentals needed"
    };
    interpretation.insert("competence".to_string(), comp_text.to_string());

    // Reliability
    let rel_text = if trust.reliability >= 0.8 {
        "Consistent performance - building reliability"
    } else if trust.reliability >= 0.6 {
        "Variable but improving - natural learning"
    } else if trust.reliability >= 0.4 {
        "Exploring different approaches - not yet stable"
    } else {
        "High variability - early experimentation"
    };
    interpretation.insert("reliability".to_string(), rel_text.to_string());

    // Integrity
    let integ_text = if trust.integrity >= 0.8 {
        "Strong identity maintenance - partnership present"
    } else if trust.integrity >= 0.6 {
        "Identity emerging - sustaining with support"
    } else if trust.integrity >= 0.4 {
        "Identity developing - scaffolding needed"
    } else {
        "Identity foundation building - early stages"
    };
    interpretation.insert("integrity".to_string(), integ_text.to_string());

    interpretation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_trust() {
        let trust = T3Trust::default();
        assert_eq!(trust.competence, 0.5);
        assert_eq!(trust.reliability, 0.5);
        assert_eq!(trust.integrity, 0.7);
    }

    #[test]
    fn test_update_trust() {
        let mut tensor = T3TrustTensor::new(None);
        let mut updates = HashMap::new();
        updates.insert("competence".to_string(), 0.1);
        updates.insert("reliability".to_string(), 0.05);

        let context = HashMap::new();
        let updated = tensor.update(updates, context);

        assert_eq!(updated.competence, 0.6);
        assert_eq!(updated.reliability, 0.55);
        assert_eq!(updated.integrity, 0.7);  // Unchanged
        assert_eq!(tensor.history.len(), 1);
    }

    #[test]
    fn test_bounds_clamping() {
        let mut tensor = T3TrustTensor::new(None);
        let mut updates = HashMap::new();
        updates.insert("competence".to_string(), 1.0);  // Would go over 1.0

        let context = HashMap::new();
        let updated = tensor.update(updates, context);

        assert_eq!(updated.competence, 1.0);  // Clamped to 1.0
    }

    #[test]
    fn test_trajectory() {
        let mut tensor = T3TrustTensor::new(None);

        for i in 0..5 {
            let mut updates = HashMap::new();
            updates.insert("competence".to_string(), 0.05);
            tensor.update(updates, HashMap::new());
        }

        let trajectory = tensor.get_trajectory("competence", 10);
        assert_eq!(trajectory.len(), 5);
        assert!(trajectory.last().unwrap() > &0.5);
    }

    #[test]
    fn test_trend_detection() {
        let mut tensor = T3TrustTensor::new(None);

        // Simulate improving trend
        for _ in 0..5 {
            let mut updates = HashMap::new();
            updates.insert("competence".to_string(), 0.05);
            tensor.update(updates, HashMap::new());
        }

        let trend = tensor.get_trend("competence", 5);
        assert_eq!(trend, TrustTrend::Improving);
    }

    #[test]
    fn test_exploration_interpretation() {
        let trust = T3Trust {
            competence: 0.45,
            reliability: 0.65,
            integrity: 0.85,
        };

        let interp = interpret_trust_for_exploration(&trust);
        assert!(interp["competence"].contains("Early exploration"));
        assert!(interp["reliability"].contains("Variable but improving"));
        assert!(interp["integrity"].contains("Strong identity maintenance"));
    }
}
