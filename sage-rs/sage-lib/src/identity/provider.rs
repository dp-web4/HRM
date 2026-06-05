use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::identity::signing::SigningContext;

const TRUST_CEILINGS: &[(&str, f64)] = &[
    ("tpm2", 1.0),
    ("tpm2_no_pcr", 0.85),
    ("fido2", 0.9),
    ("secure_enclave", 0.85),
    ("software", 0.4),
];

fn trust_ceiling_for(anchor_type: &str) -> f64 {
    TRUST_CEILINGS
        .iter()
        .find(|(k, _)| *k == anchor_type)
        .map(|(_, v)| *v)
        .unwrap_or(0.4)
}

fn now_epoch() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityManifest {
    pub name: String,
    pub lct_id: String,
    #[serde(default)]
    pub public_key_fingerprint: String,
    #[serde(default = "default_anchor")]
    pub anchor_type: String,
    #[serde(default)]
    pub machine: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub model_family: String,
    #[serde(default)]
    pub created: String,
    #[serde(default = "default_sealed_path")]
    pub sealed_path: String,
    #[serde(default = "default_trust_ceiling")]
    pub trust_ceiling: f64,
    #[serde(default = "default_status")]
    pub status: String,
}

fn default_anchor() -> String { "software".to_string() }
fn default_sealed_path() -> String { "identity.sealed".to_string() }
fn default_trust_ceiling() -> f64 { 0.4 }
fn default_status() -> String { "active".to_string() }

/// Three-layer identity provider.
/// Layer A: identity.json (public manifest)
/// Layer B: identity.sealed (encrypted root secret)
/// Layer C: identity.attest.json (attestation cache)
pub struct IdentityProvider {
    instance_dir: PathBuf,
    manifest_path: PathBuf,
    sealed_path: PathBuf,
    attest_path: PathBuf,
    manifest: Option<IdentityManifest>,
    context: Option<SigningContext>,
}

impl IdentityProvider {
    pub fn new(instance_dir: &Path) -> Self {
        let dir = instance_dir.to_path_buf();
        Self {
            manifest_path: dir.join("identity.json"),
            sealed_path: dir.join("identity.sealed"),
            attest_path: dir.join("identity.attest.json"),
            instance_dir: dir,
            manifest: None,
            context: None,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.manifest_path.exists()
    }

    pub fn is_authorized(&self) -> bool {
        self.context.is_some()
    }

    pub fn is_hardware_sealed(&self) -> bool {
        self.sealed_path.exists()
    }

    pub fn manifest(&mut self) -> Option<&IdentityManifest> {
        if self.manifest.is_none() && self.manifest_path.exists() {
            self.load_manifest();
        }
        self.manifest.as_ref()
    }

    pub fn context(&self) -> Option<&SigningContext> {
        self.context.as_ref()
    }

    /// First-time identity setup. Generates root secret, seals it, creates manifest.
    pub fn initialize(
        &mut self,
        name: &str,
        lct_id: &str,
        machine: &str,
        model: &str,
        anchor_type: &str,
    ) -> IdentityManifest {
        let secret = SigningContext::generate_secret();
        let fingerprint = SigningContext::fingerprint(&secret);

        self.seal_secret(&secret, anchor_type);

        let manifest = IdentityManifest {
            name: name.to_string(),
            lct_id: lct_id.to_string(),
            public_key_fingerprint: fingerprint.clone(),
            anchor_type: anchor_type.to_string(),
            machine: machine.to_string(),
            model: model.to_string(),
            model_family: String::new(),
            created: chrono_now_utc(),
            sealed_path: "identity.sealed".to_string(),
            trust_ceiling: trust_ceiling_for(anchor_type),
            status: "active".to_string(),
        };

        self.manifest = Some(manifest.clone());
        self.save_manifest();
        self.create_attestation(anchor_type, "enrollment");

        self.context = Some(SigningContext::new(secret, &fingerprint, anchor_type));

        manifest
    }

    /// Unseal the root secret and authorize the identity.
    pub fn authorize(&mut self) -> Option<&SigningContext> {
        if !self.is_initialized() {
            return None;
        }
        self.load_manifest();

        let secret = self.unseal_secret()?;
        let manifest = self.manifest.as_ref()?;

        self.context = Some(SigningContext::new(
            secret,
            &manifest.public_key_fingerprint,
            &manifest.anchor_type,
        ));

        self.create_attestation(&manifest.anchor_type.clone(), "session_start");
        self.context.as_ref()
    }

    /// Clear the in-memory signing context.
    pub fn lock(&mut self) {
        self.context = None;
    }

    /// Read the cached attestation (Layer C).
    pub fn get_attestation(&self) -> Option<serde_json::Value> {
        let data = std::fs::read_to_string(&self.attest_path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Load identity.json as raw JSON (legacy compatibility).
    pub fn load_legacy_state(&self) -> serde_json::Value {
        std::fs::read_to_string(&self.manifest_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
    }

    // --- Internal ---

    fn load_manifest(&mut self) {
        let data = match std::fs::read_to_string(&self.manifest_path) {
            Ok(s) => s,
            Err(_) => return,
        };

        let json: serde_json::Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(_) => return,
        };

        // Handle legacy format with nested "identity" key
        if let Some(identity) = json.get("identity") {
            let anchor = json.get("anchor_type")
                .and_then(|v| v.as_str())
                .unwrap_or("software");

            self.manifest = Some(IdentityManifest {
                name: identity.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                lct_id: identity.get("lct").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                public_key_fingerprint: identity.get("public_key_fingerprint")
                    .and_then(|v| v.as_str()).unwrap_or("").to_string(),
                anchor_type: anchor.to_string(),
                machine: identity.get("machine").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                model: identity.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                model_family: identity.get("model_family").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                created: identity.get("created").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                sealed_path: "identity.sealed".to_string(),
                trust_ceiling: trust_ceiling_for(anchor),
                status: "active".to_string(),
            });
        } else {
            self.manifest = serde_json::from_value(json).ok();
        }
    }

    fn save_manifest(&self) {
        let manifest = match &self.manifest {
            Some(m) => m,
            None => return,
        };
        let json = serde_json::to_string_pretty(manifest).unwrap_or_default();
        let _ = std::fs::write(&self.manifest_path, json);
    }

    fn seal_secret(&self, secret: &[u8], _anchor_type: &str) {
        let machine_key = self.derive_machine_key();
        let sealed: Vec<u8> = secret.iter().zip(machine_key.iter()).map(|(a, b)| a ^ b).collect();

        let mut content = b"SAGE_SEALED_v1\nsoftware\n".to_vec();
        content.extend_from_slice(&sealed);
        let _ = std::fs::write(&self.sealed_path, content);
    }

    fn unseal_secret(&self) -> Option<Vec<u8>> {
        let data = std::fs::read(&self.sealed_path).ok()?;

        // Parse header
        let first_nl = data.iter().position(|&b| b == b'\n')?;
        if &data[..first_nl] != b"SAGE_SEALED_v1" {
            return None;
        }
        let second_nl = data[first_nl + 1..].iter().position(|&b| b == b'\n')? + first_nl + 1;
        let sealed = &data[second_nl + 1..];

        let machine_key = self.derive_machine_key();
        let secret: Vec<u8> = sealed.iter().zip(machine_key.iter()).map(|(a, b)| a ^ b).collect();
        Some(secret)
    }

    fn derive_machine_key(&self) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let machine_id = format!("{}:0:{}", hostname, self.instance_dir.display());
        let hash = Sha256::digest(machine_id.as_bytes());
        hash.to_vec()
    }

    fn create_attestation(&self, anchor_type: &str, purpose: &str) {
        let entity_id = self.manifest.as_ref()
            .map(|m| m.lct_id.as_str())
            .unwrap_or("");

        let attest = serde_json::json!({
            "entity_id": entity_id,
            "anchor_type": anchor_type,
            "purpose": purpose,
            "timestamp": now_epoch(),
            "trust_ceiling": trust_ceiling_for(anchor_type),
            "version": "0.1",
        });

        let _ = std::fs::write(
            &self.attest_path,
            serde_json::to_string_pretty(&attest).unwrap_or_default(),
        );
    }
}

fn chrono_now_utc() -> String {
    let secs = now_epoch() as u64;
    let days = secs / 86400;
    let rem = secs % 86400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    // Approximate date — good enough for a timestamp string
    format!("{days}d-{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sage-id-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn full_lifecycle() {
        let dir = temp_dir();
        let mut provider = IdentityProvider::new(&dir);

        assert!(!provider.is_initialized());
        assert!(!provider.is_authorized());

        let manifest = provider.initialize("test", "lct://test", "testmachine", "model:1b", "software");
        assert!(provider.is_initialized());
        assert!(provider.is_authorized());
        assert_eq!(manifest.trust_ceiling, 0.4);
        assert!(!manifest.public_key_fingerprint.is_empty());

        // Sign something
        let sig = provider.context().unwrap().sign(b"hello");
        assert_eq!(sig.len(), 32);

        // Lock and re-authorize
        provider.lock();
        assert!(!provider.is_authorized());

        let ctx = provider.authorize().unwrap();
        let sig2 = ctx.sign(b"hello");
        assert_eq!(sig, sig2);

        // Attestation exists
        let attest = provider.get_attestation().unwrap();
        assert_eq!(attest["anchor_type"], "software");

        // Files exist
        assert!(dir.join("identity.json").exists());
        assert!(dir.join("identity.sealed").exists());
        assert!(dir.join("identity.attest.json").exists());

        cleanup(&dir);
    }

    #[test]
    fn loads_real_sprout_identity() {
        let sprout_dir = Path::new("/home/sprout/ai-workspace/SAGE/sage/instances/sprout-qwen3.5-0.8b");
        if !sprout_dir.join("identity.json").exists() {
            return; // Skip if not on Sprout
        }

        let mut provider = IdentityProvider::new(sprout_dir);
        assert!(provider.is_initialized());

        let manifest = provider.manifest().unwrap();
        assert!(!manifest.name.is_empty());
        assert!(!manifest.lct_id.is_empty());
    }
}
