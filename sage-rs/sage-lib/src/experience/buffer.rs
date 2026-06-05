use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::consciousness::observation::SalienceScore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceEntry {
    pub id: String,
    pub prompt: String,
    pub response: String,
    pub salience: SalienceScore,
    pub metabolic_state: String,
    pub atp_percentage: f64,
    pub cycle: u64,
    pub timestamp: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub machine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl ExperienceEntry {
    fn compute_id(prompt: &str, response: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(format!("{}|{}", prompt, response));
        let hash = hasher.finalize();
        hex::encode(&hash[..8])
    }

    pub fn new(
        prompt: String,
        response: String,
        salience: SalienceScore,
        metabolic_state: &str,
        atp_percentage: f64,
        cycle: u64,
    ) -> Self {
        let id = Self::compute_id(&prompt, &response);
        let timestamp = crate::snarc::temporal::now_secs();
        Self {
            id,
            prompt,
            response,
            salience,
            metabolic_state: metabolic_state.to_string(),
            atp_percentage,
            cycle,
            timestamp,
            machine: None,
            model: None,
        }
    }
}

pub struct ExperienceBuffer {
    path: PathBuf,
    salience_threshold: f64,
    recent_responses: Vec<String>,
    repetition_window: usize,
    count: u64,
}

impl ExperienceBuffer {
    pub fn new(path: &Path, salience_threshold: f64) -> Self {
        let count = Self::count_lines(path);
        Self {
            path: path.to_path_buf(),
            salience_threshold,
            recent_responses: Vec::new(),
            repetition_window: 10,
            count,
        }
    }

    pub fn with_defaults(path: &Path) -> Self {
        Self::new(path, 0.5)
    }

    fn count_lines(path: &Path) -> u64 {
        fs::read_to_string(path)
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count() as u64)
            .unwrap_or(0)
    }

    fn is_repetitive(&self, response: &str) -> bool {
        let response_words: std::collections::HashSet<&str> = response.split_whitespace().collect();
        if response_words.is_empty() {
            return true;
        }
        for recent in &self.recent_responses {
            let recent_words: std::collections::HashSet<&str> = recent.split_whitespace().collect();
            if recent_words.is_empty() {
                continue;
            }
            let intersection = response_words.intersection(&recent_words).count();
            let union = response_words.union(&recent_words).count();
            let jaccard = intersection as f64 / union as f64;
            if jaccard >= 0.85 {
                return true;
            }
        }
        false
    }

    pub fn record(&mut self, entry: ExperienceEntry) -> bool {
        if entry.salience.total < self.salience_threshold {
            return false;
        }
        if self.is_repetitive(&entry.response) {
            return false;
        }

        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let line = match serde_json::to_string(&entry) {
            Ok(s) => s,
            Err(_) => return false,
        };

        let ok = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut f| writeln!(f, "{}", line))
            .is_ok();

        if ok {
            self.count += 1;
            self.recent_responses.push(entry.response);
            if self.recent_responses.len() > self.repetition_window {
                self.recent_responses.remove(0);
            }
        }

        ok
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn load_recent(&self, n: usize) -> Vec<ExperienceEntry> {
        let data = match fs::read_to_string(&self.path) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let entries: Vec<ExperienceEntry> = data
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        let start = entries.len().saturating_sub(n);
        entries[start..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_path() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("sage-exp-test-{}-{n}.jsonl", std::process::id()))
    }

    fn make_entry(prompt: &str, response: &str, total_salience: f64) -> ExperienceEntry {
        ExperienceEntry::new(
            prompt.to_string(),
            response.to_string(),
            SalienceScore::from_components(total_salience, total_salience, total_salience, total_salience, total_salience),
            "wake",
            100.0,
            0,
        )
    }

    #[test]
    fn records_salient_experience() {
        let path = temp_path();
        let mut buf = ExperienceBuffer::with_defaults(&path);
        let entry = make_entry("hello", "world", 0.8);
        assert!(buf.record(entry));
        assert_eq!(buf.count(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_low_salience() {
        let path = temp_path();
        let mut buf = ExperienceBuffer::with_defaults(&path);
        let entry = make_entry("hello", "world", 0.2);
        assert!(!buf.record(entry));
        assert_eq!(buf.count(), 0);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_repetitive() {
        let path = temp_path();
        let mut buf = ExperienceBuffer::with_defaults(&path);
        let e1 = make_entry("q1", "the quick brown fox jumps over the lazy dog", 0.8);
        let e2 = make_entry("q2", "the quick brown fox jumps over the lazy dog", 0.8);
        assert!(buf.record(e1));
        assert!(!buf.record(e2));
        assert_eq!(buf.count(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn persistence_roundtrip() {
        let path = temp_path();
        {
            let mut buf = ExperienceBuffer::with_defaults(&path);
            buf.record(make_entry("q1", "answer one with unique content", 0.8));
            buf.record(make_entry("q2", "answer two with different words entirely", 0.9));
        }
        {
            let buf = ExperienceBuffer::with_defaults(&path);
            assert_eq!(buf.count(), 2);
            let recent = buf.load_recent(10);
            assert_eq!(recent.len(), 2);
            assert_eq!(recent[0].prompt, "q1");
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn id_is_deterministic() {
        let e1 = make_entry("hello", "world", 0.8);
        let e2 = make_entry("hello", "world", 0.5);
        assert_eq!(e1.id, e2.id);
    }
}
