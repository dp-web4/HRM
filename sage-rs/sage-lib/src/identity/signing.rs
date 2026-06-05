use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// In-memory authorized signing context. Never persisted.
pub struct SigningContext {
    secret: Vec<u8>,
    pub fingerprint: String,
    pub anchor_type: String,
}

impl SigningContext {
    pub fn new(secret: Vec<u8>, fingerprint: &str, anchor_type: &str) -> Self {
        Self {
            secret,
            fingerprint: fingerprint.to_string(),
            anchor_type: anchor_type.to_string(),
        }
    }

    /// Generate a 32-byte random secret.
    pub fn generate_secret() -> Vec<u8> {
        use rand::RngCore;
        let mut buf = vec![0u8; 32];
        rand::rng().fill_bytes(&mut buf);
        buf
    }

    /// SHA-256 fingerprint of a secret (first 16 hex chars).
    pub fn fingerprint(secret: &[u8]) -> String {
        use sha2::Digest;
        let hash = sha2::Sha256::digest(secret);
        hex::encode(&hash[..8])
    }

    /// Sign data with HMAC-SHA256. Returns 32-byte signature.
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        let mut mac = HmacSha256::new_from_slice(&self.secret)
            .expect("HMAC key length is always valid");
        mac.update(data);
        mac.finalize().into_bytes().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_deterministic() {
        let secret = vec![42u8; 32];
        let ctx = SigningContext::new(secret, "test", "software");
        let sig1 = ctx.sign(b"hello world");
        let sig2 = ctx.sign(b"hello world");
        assert_eq!(sig1, sig2);
        assert_eq!(sig1.len(), 32);
    }

    #[test]
    fn different_data_different_sig() {
        let secret = vec![42u8; 32];
        let ctx = SigningContext::new(secret, "test", "software");
        let sig1 = ctx.sign(b"hello");
        let sig2 = ctx.sign(b"world");
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn fingerprint_stable() {
        let secret = vec![0u8; 32];
        let fp1 = SigningContext::fingerprint(&secret);
        let fp2 = SigningContext::fingerprint(&secret);
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.len(), 16);
    }

    #[test]
    fn generate_secret_is_32_bytes() {
        let s = SigningContext::generate_secret();
        assert_eq!(s.len(), 32);
    }
}
