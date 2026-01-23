///! SAGE Core - Rust implementation of R6 framework and T3 trust tensors
///!
///! Provides high-performance, type-safe implementations of:
///! - R6 context management for training evaluation
///! - T3 trust tensor tracking across sessions
///!
///! Exposed to Python via PyO3 bindings for seamless integration
///! with SAGE training infrastructure.

pub mod r6;
pub mod t3;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Convert Python dict to Rust HashMap<String, serde_json::Value>
fn pydict_to_hashmap(py_dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, serde_json::Value>> {
    let mut map = HashMap::new();
    for (key, value) in py_dict.iter() {
        let key_str: String = key.extract()?;
        // For simplicity, convert all values to JSON strings then parse
        // In production, would handle each type properly
        let value_str = value.str()?.to_string();
        map.insert(key_str, serde_json::Value::String(value_str));
    }
    Ok(map)
}

/// PyO3 wrapper for R6Request
#[pyclass(name = "R6Request")]
struct PyR6Request {
    inner: r6::R6Request,
}

#[pymethods]
impl PyR6Request {
    /// Create new R6 request from exercise, context, and skill track
    #[new]
    fn new(
        exercise: &Bound<'_, PyDict>,
        session_context: &Bound<'_, PyDict>,
        skill_track: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        // Extract exercise details
        let exercise_type: String = exercise.get_item("type")?.unwrap().extract()?;
        let prompt: String = exercise.get_item("prompt")?.unwrap().extract()?;
        let expected: String = exercise
            .get_item("expected")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();

        // Extract session context
        let session_num: usize = session_context
            .get_item("session_num")?
            .unwrap()
            .extract()?;

        // Extract skill track info
        let track_id: String = skill_track.get_item("id")?.unwrap().extract()?;
        let track_name: String = skill_track.get_item("name")?.unwrap().extract()?;
        let track_desc: String = skill_track.get_item("description")?.unwrap().extract()?;

        // Build R6 components
        let mode = r6::R6Request::detect_mode(&exercise_type);

        let rules = r6::Rules {
            mode,
            mode_negotiated: true,
            success_criteria: r6::R6Request::success_criteria(&exercise_type),
            allow_clarification: true,
            allow_meta_cognitive: matches!(exercise_type.as_str(), "identity" | "uncertainty" | "clarify" | "philosophical"),
            not_in_mode: if mode == r6::OperationalMode::Conversation {
                Some(vec![
                    "do_not_refine".to_string(),
                    "do_not_format_markdown".to_string(),
                    "do_not_create_lists".to_string(),
                ])
            } else {
                None
            },
        };

        let role = r6::Role {
            lct: format!("lct:sage:training:T{:03}", session_num),
            position: if track_id.as_str() <= "B" {
                r6::TrainingRole::PracticeStudent
            } else if track_id == "C" {
                r6::TrainingRole::LearningPartner
            } else {
                r6::TrainingRole::SkillPractitioner
            },
            relationship_to: "Claude (training partner)".to_string(),
            phase: track_name.clone(),
            permissions: vec![
                "respond".to_string(),
                "clarify".to_string(),
                if track_id.as_str() >= "C" {
                    "create".to_string()
                } else {
                    "respond_only".to_string()
                },
            ],
        };

        let request = r6::Request {
            exercise_type: exercise_type.clone(),
            prompt: prompt.clone(),
            intent: format!("skill_practice_{}", exercise_type),
            expected_pattern: expected,
            parameters: HashMap::new(),
        };

        let reference = r6::Reference {
            previous_session: None,
            skill_track: track_id,
            track_description: track_desc,
            session_exercises_so_far: 0,
            recent_pattern: "developing".to_string(),
            identity_trajectory: "developing".to_string(),
        };

        let resource = r6::Resource {
            model: "Qwen2.5-0.5B-Instruct".to_string(),
            atp_budget: 50,
            context_window: 2048,
            temperature: 0.7,
            estimated_tokens: 100,
        };

        Ok(Self {
            inner: r6::R6Request {
                rules,
                role,
                request,
                reference,
                resource,
                created_at: chrono::Utc::now(),
                r6_version: "1.0".to_string(),
            },
        })
    }

    /// Evaluate a response and return R6Result
    fn evaluate(&self, response: String) -> PyResult<PyR6Result> {
        let result = r6::R6Result::evaluate(&self.inner, response);
        Ok(PyR6Result { inner: result })
    }

    /// Convert to Python dict for inspection
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        let _json = serde_json::to_value(&self.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Serialization error: {}", e))
        })?;

        // Convert JSON to Python dict (simplified)
        dict.set_item("mode", format!("{:?}", self.inner.rules.mode))?;
        dict.set_item("exercise_type", &self.inner.request.exercise_type)?;
        dict.set_item("lct", &self.inner.role.lct)?;

        Ok(dict.into())
    }
}

/// PyO3 wrapper for R6Result
#[pyclass(name = "R6Result")]
struct PyR6Result {
    inner: r6::R6Result,
}

#[pymethods]
impl PyR6Result {
    /// Get evaluation decision
    #[getter]
    fn evaluation(&self) -> String {
        format!("{:?}", self.inner.evaluation).to_lowercase()
    }

    /// Get rationale
    #[getter]
    fn rationale(&self) -> String {
        self.inner.rationale.clone()
    }

    /// Get mode match
    #[getter]
    fn mode_match(&self) -> bool {
        self.inner.mode_match
    }

    /// Get overall quality score
    #[getter]
    fn quality(&self) -> f32 {
        self.inner.quality.overall_quality
    }

    /// Get T3 updates as dict
    fn get_t3_updates(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("competence", self.inner.t3_updates.competence)?;
        dict.set_item("reliability", self.inner.t3_updates.reliability)?;
        dict.set_item("integrity", self.inner.t3_updates.integrity)?;
        Ok(dict.into())
    }

    /// Convert full result to Python dict
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);

        dict.set_item("evaluation", self.evaluation())?;
        dict.set_item("rationale", &self.inner.rationale)?;
        dict.set_item("mode_match", self.inner.mode_match)?;
        dict.set_item("quality", self.inner.quality.overall_quality)?;
        dict.set_item("has_identity_framing", self.inner.quality.has_identity_framing)?;
        dict.set_item("response", &self.inner.response)?;

        // Meta-cognitive signals
        let signals: Vec<String> = self.inner.meta_cognitive
            .iter()
            .map(|s| format!("{:?}", s).to_lowercase())
            .collect();
        dict.set_item("meta_cognitive", signals)?;

        // T3 updates
        dict.set_item("t3_updates", self.get_t3_updates(py)?)?;

        Ok(dict.into())
    }
}

/// PyO3 wrapper for T3TrustTensor
#[pyclass(name = "T3TrustTensor")]
struct PyT3TrustTensor {
    inner: t3::T3TrustTensor,
}

#[pymethods]
impl PyT3TrustTensor {
    /// Create new T3 tensor with optional initial trust
    #[new]
    #[pyo3(signature = (initial_trust=None))]
    fn new(initial_trust: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let trust = if let Some(dict) = initial_trust {
            Some(t3::T3Trust {
                competence: dict.get_item("competence")?.unwrap().extract()?,
                reliability: dict.get_item("reliability")?.unwrap().extract()?,
                integrity: dict.get_item("integrity")?.unwrap().extract()?,
            })
        } else {
            None
        };

        Ok(Self {
            inner: t3::T3TrustTensor::new(trust),
        })
    }

    /// Update trust with deltas
    fn update(
        &mut self,
        updates: &Bound<'_, PyDict>,
        context: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let mut rust_updates = HashMap::new();
        if let Ok(Some(val)) = updates.get_item("competence") {
            rust_updates.insert("competence".to_string(), val.extract()?);
        }
        if let Ok(Some(val)) = updates.get_item("reliability") {
            rust_updates.insert("reliability".to_string(), val.extract()?);
        }
        if let Ok(Some(val)) = updates.get_item("integrity") {
            rust_updates.insert("integrity".to_string(), val.extract()?);
        }

        let rust_context = pydict_to_hashmap(context)?;
        let updated = self.inner.update(rust_updates, rust_context);

        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("competence", updated.competence)?;
            dict.set_item("reliability", updated.reliability)?;
            dict.set_item("integrity", updated.integrity)?;
            Ok(dict.into())
        })
    }

    /// Get current trust values
    fn get_trust(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("competence", self.inner.trust.competence)?;
        dict.set_item("reliability", self.inner.trust.reliability)?;
        dict.set_item("integrity", self.inner.trust.integrity)?;
        Ok(dict.into())
    }

    /// Get trust summary
    fn get_summary(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = self.inner.get_summary();
        let dict = PyDict::new_bound(py);

        // Trust values
        let trust_dict = PyDict::new_bound(py);
        trust_dict.set_item("competence", summary.trust.competence)?;
        trust_dict.set_item("reliability", summary.trust.reliability)?;
        trust_dict.set_item("integrity", summary.trust.integrity)?;
        dict.set_item("trust", trust_dict)?;

        // Trends
        let trends_dict = PyDict::new_bound(py);
        for (dim, trend) in summary.trends.iter() {
            trends_dict.set_item(dim, format!("{:?}", trend).to_lowercase())?;
        }
        dict.set_item("trends", trends_dict)?;

        dict.set_item("history_length", summary.history_length)?;

        Ok(dict.into())
    }

    /// Get trend for a dimension
    fn get_trend(&self, dimension: &str) -> String {
        let trend = self.inner.get_trend(dimension, 5);
        format!("{:?}", trend).to_lowercase()
    }
}

/// Python module initialization
#[pymodule]
fn sage_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyR6Request>()?;
    m.add_class::<PyR6Result>()?;
    m.add_class::<PyT3TrustTensor>()?;

    /// Create R6 request (convenience function)
    #[pyfn(m)]
    fn create_r6_request<'py>(
        py: Python<'py>,
        exercise: &Bound<'py, PyDict>,
        session_context: &Bound<'py, PyDict>,
        skill_track: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyR6Request>> {
        Bound::new(
            py,
            PyR6Request::new(exercise, session_context, skill_track)?,
        )
    }

    /// Evaluate response (convenience function)
    #[pyfn(m)]
    fn evaluate_response<'py>(
        request: &Bound<'py, PyR6Request>,
        response: String,
    ) -> PyResult<PyR6Result> {
        request.borrow().evaluate(response)
    }

    /// Create T3 tracker (convenience function)
    #[pyfn(m)]
    #[pyo3(signature = (initial_trust=None))]
    fn create_t3_tracker<'py>(
        py: Python<'py>,
        initial_trust: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyT3TrustTensor>> {
        Bound::new(py, PyT3TrustTensor::new(initial_trust)?)
    }

    Ok(())
}
