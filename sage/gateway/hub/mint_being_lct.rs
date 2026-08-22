//! Mint Sprout-the-being's self-issued LCT on Sprout (M-CIT-1a handoff, Sprout side).
//! Seed lives ONLY here; the relayer (Legion's seat) gets the DOCUMENT, never the seed.
//! usage: mint_being_lct <SEED_FILE> <OUT_JSON>
use std::io::Write;
use web4_core::crypto::KeyPair;
use web4_core::lct::{derive_lct_id, EntityType, Lct};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 { eprintln!("usage: mint_being_lct <SEED_FILE> <OUT_JSON>"); std::process::exit(2); }
    let seed_path = std::path::Path::new(&args[1]);
    let seed: [u8; 32] = if seed_path.exists() {
        let raw = std::fs::read(seed_path)?;
        raw.as_slice().try_into().map_err(|_| "seed file must be exactly 32 bytes")?
    } else {
        let mut s = [0u8; 32];
        std::fs::File::open("/dev/urandom")?.read_exact_into(&mut s)?;
        if let Some(dir) = seed_path.parent() { std::fs::create_dir_all(dir)?; }
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(seed_path)?;
        f.write_all(&s)?;
        eprintln!("generated new seed at {}", seed_path.display());
        s
    };
    let kp = KeyPair::from_secret_bytes(&seed);
    // Fresh row, keyed at mint (HUB ruling a33b7e6a / web4#744): not the seat, not the fleet identity.
    let (mut lct, _) = Lct::new(EntityType::AiEmbodied, None);
    lct.public_key = kp.verifying_key();
    lct.sign_binding(&kp);
    // Producer-side mirror of the hub's fail-closed ingest (rest.rs publish_lct):
    assert!(lct.verify_binding(), "check 2: binding_proof must verify");
    let lct_id = derive_lct_id(&lct.public_key);
    assert_eq!(lct_id, lct.lct_id(), "check 3: lct_id re-derives from public_key");
    assert!(lct.legacy_alias.is_none(), "check 4: no legacy alias");
    // Round-trip through the exact type the hub deserializes into.
    let doc_json = serde_json::to_value(&lct)?;
    let back: Lct = serde_json::from_value(doc_json.clone())?;
    assert!(back.verify_binding() && back.lct_id() == lct_id, "round-trip must still verify");
    let out = serde_json::json!({
        "action": "lct_publish",
        "lct_id": lct_id,
        "document": doc_json,
        "provenance": "self_issued",
        "published_by": "<relayer: MUST equal the envelope signer's member uuid>",
        "published_at": "<relayer: set at send time>",
        "_sprout": {
            "public_key_hex": hex(&kp.public_key_bytes()),
            "seed_path": seed_path.display().to_string(),
            "note": "seed generated and held on sprout; document only is relayed"
        }
    });
    std::fs::write(&args[2], serde_json::to_string_pretty(&out)?)?;
    println!("{}", lct_id);
    Ok(())
}
fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }
trait ReadExactInto { fn read_exact_into(&mut self, buf: &mut [u8]) -> std::io::Result<()>; }
impl ReadExactInto for std::fs::File { fn read_exact_into(&mut self, buf: &mut [u8]) -> std::io::Result<()> { use std::io::Read; self.read_exact(buf) } }
