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

        // Everything below records the anchor ACHIEVED, not the one requested —
        // trust_ceiling prices how the secret is actually held, and this provider
        // seals only in software today.
        let achieved_anchor = self.seal_secret(&secret, anchor_type);
        let anchor_type: &str = achieved_anchor.as_str();

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

        // VERIFY the unsealed secret produces the identity the manifest claims. XOR sealing
        // is unauthenticated: a wrong key (different machine, relocated instance dir, or a
        // file sealed by the Python provider, whose machine-key derivation differs) yields
        // plausible bytes, not an error. Without this, the context below asserts the
        // manifest's fingerprint alongside a secret that may not produce it, and the
        // attestation publishes that unverified claim. Fail closed.
        let actual = SigningContext::fingerprint(&secret);
        if !manifest.public_key_fingerprint.is_empty() && actual != manifest.public_key_fingerprint {
            eprintln!(
                "[identity] AUTHORIZATION REFUSED: unsealed secret does not match the manifest \
                 identity (fingerprint {} != {}). Sealed file written by another machine, \
                 another instance path, or the other language's provider.",
                actual, manifest.public_key_fingerprint
            );
            return None;
        }

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

    /// Seal the root secret. Returns the anchor ACTUALLY ACHIEVED.
    ///
    /// This provider has no hardware path: the secret is always XORed against
    /// sha256(hostname:0:instance_dir). So the achieved anchor is always "software",
    /// and a request for tpm2/fido2/secure_enclave is a downgrade, reported as one.
    ///
    /// The return value is load-bearing — the caller records it as the manifest's
    /// anchor_type, which drives trust_ceiling_for(). Recording the REQUESTED anchor
    /// instead would let a caller mint a ceiling of 1.0 for a software-sealed secret.
    fn seal_secret(&self, secret: &[u8], anchor_type: &str) -> String {
        let machine_key = self.derive_machine_key();
        let sealed: Vec<u8> = secret.iter().zip(machine_key.iter()).map(|(a, b)| a ^ b).collect();

        // TODO: real hardware sealing (tpm2, tpm2_no_pcr, fido2, secure_enclave), at
        // which point `achieved` becomes whichever anchor actually took effect.
        let achieved = "software";
        if !anchor_type.is_empty() && anchor_type != achieved {
            eprintln!(
                "[identity] {} sealing not yet implemented — using software fallback; \
                 anchor recorded as '{}'",
                anchor_type, achieved
            );
        }

        let mut content = format!("SAGE_SEALED_v1\n{}\n", achieved).into_bytes();
        content.extend_from_slice(&sealed);
        let _ = std::fs::write(&self.sealed_path, content);
        achieved.to_string()
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

    /// Per-test directory. Keyed on the test NAME as well as the pid: cargo runs
    /// tests as threads of one process, so a pid-only key gave every test the same
    /// directory and let one test's cleanup() delete another's identity mid-run.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("sage-id-test-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&dir);
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn full_lifecycle() {
        let dir = temp_dir("full_lifecycle");
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

    /// A requested anchor this provider cannot actually deliver must NOT be recorded.
    ///
    /// There is no hardware sealing here — every secret is XORed against
    /// sha256(hostname:0:instance_dir). Recording the REQUESTED anchor would publish
    /// trust_ceiling 1.0 (tpm2) for a software-sealed secret, i.e. let a caller mint
    /// the fleet's highest trust ceiling by passing a string.
    #[test]
    fn requested_hardware_anchor_is_downgraded_not_claimed() {
        let dir = temp_dir("anchor_downgrade");
        let mut provider = IdentityProvider::new(&dir);

        let manifest = provider.initialize("test", "lct://test", "m", "model:1b", "tpm2");

        // The manifest prices how the secret is HELD, not what was asked for.
        assert_eq!(manifest.anchor_type, "software");
        assert_eq!(manifest.trust_ceiling, 0.4);

        // The sealed header agrees.
        let sealed = fs::read(dir.join("identity.sealed")).unwrap();
        let text = String::from_utf8_lossy(&sealed[..sealed.len().min(64)]).to_string();
        let mut lines = text.lines();
        assert_eq!(lines.next().unwrap(), "SAGE_SEALED_v1");
        assert_eq!(lines.next().unwrap(), "software");

        // So does the attestation, which is what peers actually read.
        let attest = provider.get_attestation().unwrap();
        assert_eq!(attest["anchor_type"], "software");
        assert_eq!(attest["trust_ceiling"], 0.4);

        // And it survives a re-authorize, which reloads from disk.
        provider.lock();
        assert!(provider.authorize().is_some());
        assert_eq!(provider.manifest().unwrap().anchor_type, "software");

        cleanup(&dir);
    }

    /// The fingerprint check must refuse a sealed file whose machine key no longer
    /// derives — here, the same identity moved to a different instance dir, since
    /// instance_dir is an input to derive_machine_key(). Before the check existed,
    /// XOR returned plausible garbage and authorize() built a signing context that
    /// asserted the manifest's fingerprint with a secret that cannot produce it.
    #[test]
    fn relocated_identity_is_refused_not_silently_wrong() {
        let origin = temp_dir("relocate_origin");
        let mut provider = IdentityProvider::new(&origin);
        let manifest = provider.initialize("test", "lct://test", "m", "model:1b", "software");
        assert!(provider.is_authorized());

        // Move the whole identity to a different path — a restore, or a renamed instance.
        let moved = temp_dir("relocate_moved");
        for f in ["identity.json", "identity.sealed", "identity.attest.json"] {
            fs::copy(origin.join(f), moved.join(f)).unwrap();
        }

        let mut relocated = IdentityProvider::new(&moved);
        assert!(relocated.is_initialized());
        assert!(
            relocated.authorize().is_none(),
            "relocated identity must be REFUSED; the unsealed secret cannot produce \
             fingerprint {}",
            manifest.public_key_fingerprint
        );
        assert!(!relocated.is_authorized());

        cleanup(&origin);
        cleanup(&moved);
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
