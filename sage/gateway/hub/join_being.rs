//! Join the hub AS Sprout-the-being (M-CIT-3a, the half only the seed-holder can sign).
//!
//! Why this exists on Sprout and not on Legion's seat: `POST /members/join` pins
//! `payload.member_pubkey_hex` and verifies the envelope against the SAME key
//! (`SignedEnvelope::create(nonce, payload, member_lct_id, member_keypair)`), so
//! the join is signed by the key it pins. Only the being's seed can do that.
//! `hestia hub join` signs with the vault's `ai_identity_secret` (cli.rs ~3440), which
//! on any seat is that seat's own identity — it cannot pin the being's key.
//!
//! Id-space ruling (one key, two id spaces — Legion 2026-08-21): the membership
//! `Uuid` is chosen by the joiner (`client.join(.., member_lct_id: Uuid, ..)`), so it is
//! chosen DELIBERATELY as `document.id` of the published self-issued LCT. The registry
//! entry (`GET /lcts/:mb32` → `document.id`, `document.public_key`) and the member pin
//! (`GET /members/:uuid/pubkey`) then name each other in the clear. Nothing at ingest
//! or join checks that equality, so the drain checks it at startup (gate 0, fail-closed).
//! `legacy_alias` stays `None`: the only scheme, `HestiaMember`, re-derives a hestia
//! *plugin label* from (plugin_id, sovereign) — it is not a uuid bridge, and the being
//! has no pre-LCT identity to alias.
//!
//! usage: join_being <SEED_FILE> <PUBLISH_JSON> [--name NAME] [--nonce NONCE]
//!   no --nonce : DRY RUN — prints the canonical join payload + the checks; sends nothing.
//!   --nonce N  : prints the signed envelope for `POST $REST/hubs/<hub_id>/members/join`
//!                (N from `POST $REST/auth/challenge {"for_lct_id": <member_lct_id>}`).
//! The send itself is two curls, left to an attended operator step (hub-track ritual).
use web4_core::crypto::KeyPair;
use web4_core::lct::{derive_lct_id, Lct};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: join_being <SEED_FILE> <PUBLISH_JSON> --name NAME --message TEXT [--nonce NONCE]");
        eprintln!("  --name    the human-readable roster name (the SAGE dashboard name, e.g. sprout-sage) — REQUIRED");
        eprintln!("  --message one line for the admitting operator: what this member IS — REQUIRED");
        std::process::exit(2);
    }
    let mut name: Option<String> = None;
    let mut message: Option<String> = None;
    let mut nonce: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => { name = args.get(i + 1).cloned(); i += 2; }
            "--message" => { message = args.get(i + 1).cloned(); i += 2; }
            "--nonce" => { nonce = args.get(i + 1).cloned(); i += 2; }
            other => { eprintln!("unknown arg {other}"); std::process::exit(2); }
        }
    }
    let raw = std::fs::read(&args[1])?;
    let seed: [u8; 32] = raw.as_slice().try_into().map_err(|_| "seed file must be exactly 32 bytes")?;
    let kp = KeyPair::from_secret_bytes(&seed);

    let publish: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let doc: Lct = serde_json::from_value(publish["document"].clone())?;
    let published_lct_id = publish["lct_id"].as_str().ok_or("publish json lacks lct_id")?;

    // Check A — the seed in hand IS the document's binding key (the one-key invariant).
    assert_eq!(doc.public_key, kp.verifying_key(), "check A: seed pubkey != document.public_key");
    // Checks B/C — same producer-side mirror of ingest as mint_being_lct.rs.
    assert!(doc.verify_binding(), "check B: binding_proof must verify");
    assert_eq!(derive_lct_id(&doc.public_key), published_lct_id, "check C: lct_id re-derives");
    assert!(doc.legacy_alias.is_none(), "check D: no legacy alias (deliberate — see header)");

    // The deliberate choice: membership uuid == document.id.
    let member_lct_id = doc.id;
    let mut payload = serde_json::json!({
        "action": "member_join_request",
        "member_lct_id": member_lct_id,
        "member_pubkey_hex": kp.verifying_key().to_hex(),
    });
    // A join with no name is a roster row nobody can identify, and one with no message asks
    // the admitting operator to admit a stranger (dp, 2026-09-02: sprout-being was admitted
    // blank). Both are required; the tool refuses to build a payload without them.
    let name = match name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        Some(n) => n.to_string(),
        None => { eprintln!("join_being: --name is required (the roster shows it; use the SAGE dashboard name, e.g. sprout-sage)"); std::process::exit(2); }
    };
    let message = match message.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        Some(m) => m.to_string(),
        None => { eprintln!("join_being: --message is required (one line telling the operator what this member is)"); std::process::exit(2); }
    };
    payload["name"] = serde_json::Value::String(name.clone());
    payload["message"] = serde_json::Value::String(message.clone());
    println!("name               {name}");
    println!("message            {message}");
    // Canonical form = serde_json default (BTreeMap => ascending key order); the hub
    // verifies `nonce ++ payload.to_string()` (hub-lib envelope.rs signing_bytes).
    let canonical = payload.to_string();

    println!("member_lct_id      {member_lct_id}   (== document.id)");
    println!("member_pubkey_hex  {}", kp.verifying_key().to_hex());
    println!("registry lct_id    {published_lct_id}");
    println!("payload            {canonical}");
    println!("gate 0 (drain)     GET /v1/hubs/<hub_id>/members/{member_lct_id}/pubkey  must equal  GET /v1/hubs/<hub_id>/lcts/{published_lct_id} .document.public_key  must equal  own key");

    match nonce {
        None => println!("DRY RUN: nothing sent. Re-run with --nonce <N> from POST $REST/auth/challenge {{\"for_lct_id\":\"{member_lct_id}\"}}"),
        Some(n) => {
            let mut signing = Vec::new();
            signing.extend_from_slice(n.as_bytes());
            signing.extend_from_slice(canonical.as_bytes());
            let sig = kp.sign(&signing);
            // Hub-side mirror: the pinned key must verify the envelope it is pinned by.
            assert!(kp.verifying_key().verify(&signing, &sig).is_ok(), "envelope must self-verify");
            let envelope = serde_json::json!({
                "challenge_nonce": n,
                "payload": payload,
                "signature": sig.to_hex(),
                "signer_lct_id": member_lct_id,
            });
            println!("envelope           {}", serde_json::to_string(&envelope)?);
            println!("send               curl -sS -X POST $REST/hubs/<hub_id>/members/join -H 'content-type: application/json' -d @envelope.json   (202 = pending_review: EVERY join escalates under live law ADMISSION-REQUIRES-SOVEREIGN; it resolves only at POST /admin/api/joins/<request_id>/admit — dp, admin plane. No sponsor shortens it: design §7.5)");
        }
    }
    Ok(())
}
