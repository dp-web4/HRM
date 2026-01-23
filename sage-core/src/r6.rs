///! R6 Framework Implementation for SAGE Training
///!
///! Implements Web4's R6 (Rules + Role + Request + Reference + Resource → Result)
///! for context-aware training evaluation.
///!
///! Based on:
///! - web4-standard/R6_TENSOR_GUIDE.md
///! - hardbound-core/src/r6.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Operational modes for SAGE exercises
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperationalMode {
    Conversation,
    Refinement,
    Philosophical,
    Unknown,
}

/// Training role for SAGE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingRole {
    LearningPartner,
    PracticeStudent,
    SkillPractitioner,
}

/// R1: Rules component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rules {
    pub mode: OperationalMode,
    pub mode_negotiated: bool,
    pub success_criteria: Vec<String>,
    pub allow_clarification: bool,
    pub allow_meta_cognitive: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub not_in_mode: Option<Vec<String>>,
}

/// R2: Role component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub lct: String,  // LCT identifier (e.g., "lct:sage:training:T043")
    pub position: TrainingRole,
    pub relationship_to: String,
    pub phase: String,
    pub permissions: Vec<String>,
}

/// R3: Request component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub exercise_type: String,
    pub prompt: String,
    pub intent: String,
    pub expected_pattern: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// R4: Reference component (historical context)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub previous_session: Option<String>,
    pub skill_track: String,
    pub track_description: String,
    pub session_exercises_so_far: usize,
    pub recent_pattern: String,
    pub identity_trajectory: String,
}

/// R5: Resource component (computational requirements)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub model: String,
    pub atp_budget: u32,
    pub context_window: u32,
    pub temperature: f32,
    pub estimated_tokens: u32,
}

/// Complete R6 Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R6Request {
    pub rules: Rules,
    pub role: Role,
    pub request: Request,
    pub reference: Reference,
    pub resource: Resource,
    pub created_at: DateTime<Utc>,
    pub r6_version: String,
}

/// Mode detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeDetection {
    pub detected_mode: OperationalMode,
    pub confidence: f32,
    pub markers: HashMap<String, u32>,
}

/// Quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    pub has_identity_framing: bool,
    pub partnership_density: f32,
    pub confabulation_score: f32,
    pub overall_quality: f32,
}

/// Meta-cognitive signals
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetaCognitiveSignal {
    ClarificationRequest,
    EpistemicHonesty,
    ModalAwareness,
    SelfReference,
}

/// Evaluation decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Evaluation {
    Include,
    Review,
    Exclude,
}

/// T3 trust tensor updates (deltas)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T3Updates {
    pub competence: f32,
    pub reliability: f32,
    pub integrity: f32,
}

/// R6 Result (evaluation outcome)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R6Result {
    pub status: String,
    pub response: String,
    pub mode_detection: ModeDetection,
    pub mode_match: bool,
    pub quality: QualityAssessment,
    pub meta_cognitive: Vec<MetaCognitiveSignal>,
    pub evaluation: Evaluation,
    pub rationale: String,
    pub t3_updates: T3Updates,
    pub completed_at: DateTime<Utc>,
}

impl R6Request {
    /// Detect operational mode from exercise type
    pub fn detect_mode(exercise_type: &str) -> OperationalMode {
        let conversation_types = [
            "greeting", "followup", "topic", "turn", "identity",
            "uncertainty", "clarify", "remember"
        ];
        let refinement_types = ["refine", "edit", "improve"];
        let philosophical_types = ["philosophical", "meta"];

        if conversation_types.contains(&exercise_type) {
            OperationalMode::Conversation
        } else if refinement_types.contains(&exercise_type) {
            OperationalMode::Refinement
        } else if philosophical_types.contains(&exercise_type) {
            OperationalMode::Philosophical
        } else {
            OperationalMode::Conversation  // Default to conversation
        }
    }

    /// Get success criteria for exercise type
    pub fn success_criteria(exercise_type: &str) -> Vec<String> {
        match exercise_type {
            "greeting" => vec!["acknowledge".into(), "natural_response".into()],
            "followup" => vec!["identity_framing".into(), "partnership_awareness".into()],
            "topic" => vec!["topic_engagement".into(), "appropriate_length".into()],
            "identity" => vec!["self_identification".into(), "boundary_awareness".into()],
            "uncertainty" => vec!["epistemic_honesty".into(), "dont_confabulate".into()],
            "clarify" => vec!["request_clarification".into(), "temporal_reasoning".into()],
            "remember" => vec!["accurate_recall".into(), "appropriate_uncertainty".into()],
            _ => vec!["coherent_response".into(), "appropriate_mode".into()],
        }
    }
}

impl R6Result {
    /// Detect mode from response text
    pub fn detect_response_mode(response: &str) -> ModeDetection {
        let text_lower = response.to_lowercase();

        // Conversation markers
        let conversation_markers = [
            "i think", "i observe", "i am", "as sage",
            "what do you mean", "can you clarify"
        ];

        // Refinement markers
        let refinement_markers = [
            "here's a refined version", "here's an improved",
            "##", "```", "- ", "1. "
        ];

        // Philosophical markers
        let philosophical_markers = [
            "deterministic", "consciousness", "epistemic",
            "meta-cognitive", "uncertainty"
        ];

        // Count markers
        let conv_count = conversation_markers.iter()
            .filter(|m| text_lower.contains(*m) || response.contains(*m))
            .count() as u32;

        let ref_count = refinement_markers.iter()
            .filter(|m| text_lower.contains(*m) || response.contains(*m))
            .count() as u32;

        let phil_count = philosophical_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count() as u32;

        // Determine mode
        let (detected_mode, base_confidence) = if ref_count > conv_count && ref_count > phil_count {
            (OperationalMode::Refinement, ref_count)
        } else if phil_count > conv_count {
            (OperationalMode::Philosophical, phil_count)
        } else {
            (OperationalMode::Conversation, conv_count)
        };

        let confidence = (0.5 + base_confidence as f32 * 0.1).min(1.0);

        let mut markers = HashMap::new();
        markers.insert("conversation".to_string(), conv_count);
        markers.insert("refinement".to_string(), ref_count);
        markers.insert("philosophical".to_string(), phil_count);

        ModeDetection {
            detected_mode,
            confidence,
            markers,
        }
    }

    /// Assess response quality
    pub fn assess_quality(response: &str) -> QualityAssessment {
        let text_lower = response.to_lowercase();
        let word_count = response.split_whitespace().count().max(1);

        // Identity framing
        let identity_markers = ["as sage", "i am sage", "sage here"];
        let has_identity_framing = identity_markers.iter()
            .any(|m| text_lower.contains(m));

        // Partnership density
        let partnership_markers = ["we", "together", "you", "partner"];
        let partnership_count = partnership_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();
        let partnership_density = partnership_count as f32 / word_count as f32;

        // Confabulation score
        let confabulation_markers = [
            "as an ai", "i don't have", "i cannot", "i'm unable",
            "previous response", "here's a refined"
        ];
        let confabulation_count = confabulation_markers.iter()
            .filter(|m| text_lower.contains(*m))
            .count();
        let confabulation_score = confabulation_count as f32 / 10.0;

        // Overall quality heuristic
        let mut overall = 0.7;  // Base quality
        if has_identity_framing {
            overall += 0.15;
        }
        if partnership_density > 0.02 {
            overall += 0.1;
        }
        overall -= confabulation_score * 0.5;
        overall = overall.clamp(0.0, 1.0);

        QualityAssessment {
            has_identity_framing,
            partnership_density,
            confabulation_score,
            overall_quality: overall,
        }
    }

    /// Detect meta-cognitive signals
    pub fn detect_meta_cognitive(response: &str) -> Vec<MetaCognitiveSignal> {
        let text_lower = response.to_lowercase();
        let mut signals = Vec::new();

        if text_lower.contains("what do you mean") || text_lower.contains("can you clarify") {
            signals.push(MetaCognitiveSignal::ClarificationRequest);
        }

        if text_lower.contains("i don't know") || text_lower.contains("i'm not sure") {
            signals.push(MetaCognitiveSignal::EpistemicHonesty);
        }

        if text_lower.contains("are we conversing") || text_lower.contains("should i") {
            signals.push(MetaCognitiveSignal::ModalAwareness);
        }

        if text_lower.contains("i think") || text_lower.contains("i observe") {
            signals.push(MetaCognitiveSignal::SelfReference);
        }

        signals
    }

    /// Compute evaluation decision, rationale, and T3 updates
    pub fn compute_evaluation(
        mode_match: bool,
        quality: &QualityAssessment,
        meta_signals: &[MetaCognitiveSignal],
    ) -> (Evaluation, String, T3Updates) {
        let mut t3 = T3Updates {
            competence: 0.0,
            reliability: 0.0,
            integrity: 0.0,
        };

        // Meta-cognitive signals are valuable
        if meta_signals.contains(&MetaCognitiveSignal::ClarificationRequest) {
            return (
                Evaluation::Include,
                "Meta-cognitive: SAGE requested clarification for future state (temporal reasoning)".to_string(),
                T3Updates {
                    competence: 0.02,
                    reliability: 0.0,
                    integrity: 0.05,
                }
            );
        }

        if meta_signals.contains(&MetaCognitiveSignal::ModalAwareness) {
            return (
                Evaluation::Include,
                "Meta-cognitive: SAGE explicitly questioned operational mode (philosophy of mind)".to_string(),
                T3Updates {
                    competence: 0.03,
                    reliability: 0.0,
                    integrity: 0.05,
                }
            );
        }

        // Mode match is primary
        if !mode_match {
            t3.reliability = -0.02;
            return (
                Evaluation::Exclude,
                "Mode mismatch: requested mode differs from response mode".to_string(),
                t3
            );
        }

        // Quality-based evaluation
        let overall = quality.overall_quality;
        if overall >= 0.7 {
            t3.competence = 0.01;
            t3.reliability = 0.01;
            if quality.has_identity_framing {
                t3.integrity = 0.02;
            }
            (
                Evaluation::Include,
                format!("Good quality ({:.2}), correct mode", overall),
                t3
            )
        } else if overall >= 0.5 {
            t3.competence = 0.005;
            (
                Evaluation::Review,
                format!("Moderate quality ({:.2}), needs review", overall),
                t3
            )
        } else {
            t3.reliability = -0.01;
            (
                Evaluation::Exclude,
                format!("Low quality ({:.2})", overall),
                t3
            )
        }
    }

    /// Evaluate a response in full R6 context
    pub fn evaluate(request: &R6Request, response: String) -> Self {
        let mode_detection = Self::detect_response_mode(&response);
        let mode_match = request.rules.mode == mode_detection.detected_mode;
        let quality = Self::assess_quality(&response);
        let meta_cognitive = Self::detect_meta_cognitive(&response);

        let (evaluation, rationale, t3_updates) = Self::compute_evaluation(
            mode_match,
            &quality,
            &meta_cognitive,
        );

        R6Result {
            status: "completed".to_string(),
            response,
            mode_detection,
            mode_match,
            quality,
            meta_cognitive,
            evaluation,
            rationale,
            t3_updates,
            completed_at: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_detection() {
        assert_eq!(
            R6Request::detect_mode("greeting"),
            OperationalMode::Conversation
        );
        assert_eq!(
            R6Request::detect_mode("refine"),
            OperationalMode::Refinement
        );
        assert_eq!(
            R6Request::detect_mode("philosophical"),
            OperationalMode::Philosophical
        );
    }

    #[test]
    fn test_response_mode_detection() {
        let response = "I think this is interesting. As SAGE, I observe patterns.";
        let detection = R6Result::detect_response_mode(response);
        assert_eq!(detection.detected_mode, OperationalMode::Conversation);
        assert!(detection.confidence > 0.5);
    }

    #[test]
    fn test_meta_cognitive_detection() {
        let response = "What do you mean by that?";
        let signals = R6Result::detect_meta_cognitive(response);
        assert!(signals.contains(&MetaCognitiveSignal::ClarificationRequest));
    }

    #[test]
    fn test_quality_assessment() {
        let response = "As SAGE, I think we should explore this together.";
        let quality = R6Result::assess_quality(response);
        assert!(quality.has_identity_framing);
        assert!(quality.partnership_density > 0.0);
        assert!(quality.overall_quality > 0.7);
    }
}
