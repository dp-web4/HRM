use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

use sage_lib::federation::fleet::FleetRegistry;
use sage_lib::federation::peer_trust::PeerTrustTracker;

use super::monitor::PeerState;

#[derive(Serialize)]
struct PeerChatRequest<'a> {
    sender: &'a str,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    conversation_id: Option<&'a str>,
}

#[derive(Deserialize)]
pub struct PeerChatResponse {
    #[serde(default)]
    pub success: bool,
    #[serde(default)]
    pub response: String,
    #[serde(default)]
    pub metabolic_state: Option<String>,
    #[serde(default)]
    pub atp_remaining: Option<f64>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct PeerClient {
    fleet: FleetRegistry,
    trust: Arc<Mutex<PeerTrustTracker>>,
    peer_states: Arc<Mutex<HashMap<String, PeerState>>>,
    client: Client,
    self_name: String,
    messages_sent: Arc<Mutex<u64>>,
    messages_failed: Arc<Mutex<u64>>,
}

pub struct SendResult {
    pub response: String,
    pub metabolic_state: Option<String>,
    pub atp_remaining: Option<f64>,
    pub latency_ms: f64,
}

impl PeerClient {
    pub fn new(
        fleet: FleetRegistry,
        trust: Arc<Mutex<PeerTrustTracker>>,
        peer_states: Arc<Mutex<HashMap<String, PeerState>>>,
        self_name: &str,
    ) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        Self {
            fleet,
            trust,
            peer_states,
            client,
            self_name: self_name.to_string(),
            messages_sent: Arc::new(Mutex::new(0)),
            messages_failed: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn send_message(
        &self,
        target: &str,
        message: &str,
        conversation_id: Option<&str>,
    ) -> Result<SendResult, String> {
        let url = self.fleet.gateway_url(target)
            .ok_or_else(|| format!("unknown peer: {}", target))?;

        let states = self.peer_states.lock().await;
        if let Some(state) = states.get(target) {
            if !state.online {
                warn!("sending to offline peer {target} (last error: {:?})", state.error);
            }
        }
        drop(states);

        let req = PeerChatRequest {
            sender: &self.self_name,
            message,
            conversation_id,
        };

        let chat_url = format!("{}/chat", url);
        let start = Instant::now();

        let result = self.client
            .post(&chat_url)
            .json(&req)
            .send()
            .await;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<PeerChatResponse>().await {
                    Ok(peer_resp) => {
                        let mut trust = self.trust.lock().await;
                        trust.record_interaction(target, "success");
                        drop(trust);

                        *self.messages_sent.lock().await += 1;

                        info!("message to {target}: {:.0}ms", latency_ms);

                        Ok(SendResult {
                            response: peer_resp.response,
                            metabolic_state: peer_resp.metabolic_state,
                            atp_remaining: peer_resp.atp_remaining,
                            latency_ms,
                        })
                    }
                    Err(e) => {
                        let mut trust = self.trust.lock().await;
                        trust.record_interaction(target, "error");
                        *self.messages_failed.lock().await += 1;
                        Err(format!("parse error from {target}: {e}"))
                    }
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let mut trust = self.trust.lock().await;
                trust.record_interaction(target, "error");
                *self.messages_failed.lock().await += 1;
                Err(format!("{target} returned HTTP {status}"))
            }
            Err(e) => {
                let mut trust = self.trust.lock().await;
                trust.record_interaction(target, "timeout");
                *self.messages_failed.lock().await += 1;
                Err(format!("failed to reach {target}: {e}"))
            }
        }
    }

    pub async fn stats(&self) -> (u64, u64) {
        let sent = *self.messages_sent.lock().await;
        let failed = *self.messages_failed.lock().await;
        (sent, failed)
    }
}
