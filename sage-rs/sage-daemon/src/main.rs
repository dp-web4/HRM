mod ollama;
mod consciousness;

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::info;

use sage_lib::experience::buffer::ExperienceBuffer;
use sage_lib::federation::fleet::FleetRegistry;
use sage_lib::metabolic::controller::{CycleData, MetabolicController, MetabolicState};
use sage_lib::snarc::arousal::ArousalDetector;
use sage_lib::snarc::conflict::ConflictDetector;
use sage_lib::snarc::novelty::NoveltyDetector;
use sage_lib::snarc::reward::RewardEstimator;
use sage_lib::snarc::surprise::SurpriseDetector;
use sage_lib::snarc::temporal;

use crate::consciousness::{ConsciousnessHandle, ConsciousnessLoop};
use crate::ollama::client::{ChatMessage, OllamaClient};

struct AppState {
    surprise: Mutex<SurpriseDetector>,
    novelty: Mutex<NoveltyDetector>,
    arousal: Mutex<ArousalDetector>,
    reward: Mutex<RewardEstimator>,
    conflict: Mutex<ConflictDetector>,
    metabolic: Mutex<MetabolicController>,
    consciousness: ConsciousnessHandle,
    ollama: OllamaClient,
    fleet: Option<FleetRegistry>,
    started: Instant,
    model: String,
}

// --- Request/Response types ---

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    uptime_secs: f64,
    version: &'static str,
    port: u16,
    model: String,
    ollama_available: bool,
    consciousness_loop: bool,
}

#[derive(Serialize)]
struct StatusResponse {
    daemon: &'static str,
    sprint: &'static str,
    snarc_detectors: Vec<&'static str>,
    half_lives: Vec<(&'static str, f64)>,
    metabolic_state: &'static str,
    atp_percentage: f64,
    total_cycles: u64,
    model: String,
    fleet_size: usize,
    consciousness_loop: bool,
}

#[derive(Deserialize)]
struct SnarcRequest {
    observation: f64,
    sensor_id: Option<String>,
}

#[derive(Serialize)]
struct SnarcResponse {
    surprise: f64,
    novelty: f64,
    arousal: f64,
    reward: f64,
    sensor_id: String,
}

#[derive(Deserialize)]
struct MetabolicCycleRequest {
    max_salience: Option<f64>,
    crisis_detected: Option<bool>,
}

#[derive(Serialize)]
struct MetabolicResponse {
    state: &'static str,
    atp_current: f64,
    atp_percentage: f64,
    total_cycles: u64,
    transitions: usize,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    conversation: Option<Vec<ChatMessage>>,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
    model: String,
    metabolic_state: String,
    atp_percentage: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    salience: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cycle: Option<u64>,
}

#[derive(Deserialize)]
struct StreamRequest {
    message: String,
    #[serde(default)]
    system: Option<String>,
}

#[derive(Serialize)]
struct PeerInfo {
    name: String,
    gateway_url: String,
    pool: String,
    model: String,
    device: String,
    hardware: String,
}

#[derive(Serialize)]
struct PeersResponse {
    self_machine: String,
    peers: Vec<PeerInfo>,
    fleet_version: u32,
}

const PORT: u16 = 8760;
const FLEET_JSON: &str = "/home/sprout/ai-workspace/SAGE/sage/federation/fleet.json";
const EXPERIENCE_PATH: &str = "/home/sprout/ai-workspace/SAGE/sage/instances/sprout-qwen3.5-0.8b/experience_buffer_rs.jsonl";

// --- Handlers ---

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let available = state.ollama.is_available().await;
    Json(HealthResponse {
        status: "ok",
        uptime_secs: state.started.elapsed().as_secs_f64(),
        version: env!("CARGO_PKG_VERSION"),
        port: PORT,
        model: state.model.clone(),
        ollama_available: available,
        consciousness_loop: true,
    })
}

async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    let ctrl = state.metabolic.lock().await;
    Json(StatusResponse {
        daemon: "sage-daemon",
        sprint: "4 — consciousness loop",
        snarc_detectors: vec!["surprise", "novelty", "arousal", "reward", "conflict"],
        half_lives: temporal::DEFAULT_HALF_LIVES.to_vec(),
        metabolic_state: ctrl.current_state.as_str(),
        atp_percentage: ctrl.atp_percentage(),
        total_cycles: ctrl.total_cycles,
        model: state.model.clone(),
        fleet_size: state.fleet.as_ref().map_or(0, |f| f.fleet_size()),
        consciousness_loop: true,
    })
}

async fn snarc_observe(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SnarcRequest>,
) -> Json<SnarcResponse> {
    let sensor_id = req.sensor_id.unwrap_or_else(|| "default".to_string());
    let obs = req.observation;

    let surprise = state.surprise.lock().await.compute(obs, &sensor_id);
    let novelty = state.novelty.lock().await.compute(obs, &sensor_id);
    let arousal = state.arousal.lock().await.compute(obs, &sensor_id);
    let reward = state.reward.lock().await.compute(obs, &sensor_id);

    Json(SnarcResponse { surprise, novelty, arousal, reward, sensor_id })
}

async fn metabolic_cycle(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MetabolicCycleRequest>,
) -> Json<MetabolicResponse> {
    let data = CycleData {
        max_salience: req.max_salience.unwrap_or(0.0),
        crisis_detected: req.crisis_detected.unwrap_or(false),
        ..Default::default()
    };
    let mut ctrl = state.metabolic.lock().await;
    ctrl.update(&data);
    Json(MetabolicResponse {
        state: ctrl.current_state.as_str(),
        atp_current: ctrl.atp_current,
        atp_percentage: ctrl.atp_percentage(),
        total_cycles: ctrl.total_cycles,
        transitions: ctrl.history.len(),
    })
}

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    if let Some(conversation) = req.conversation {
        let mut messages = conversation;
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: req.message,
        });
        match state.ollama.chat(&messages).await {
            Ok(text) => {
                let ctrl = state.metabolic.lock().await;
                Ok(Json(ChatResponse {
                    response: text,
                    model: state.model.clone(),
                    metabolic_state: ctrl.current_state.as_str().to_string(),
                    atp_percentage: ctrl.atp_percentage(),
                    salience: None,
                    cycle: None,
                }))
            }
            Err(e) => Err((StatusCode::BAD_GATEWAY, e)),
        }
    } else {
        match state.consciousness.send_message(req.message, req.system, "http").await {
            Ok(resp) => Ok(Json(ChatResponse {
                response: resp.text,
                model: state.model.clone(),
                metabolic_state: resp.metabolic_state,
                atp_percentage: resp.atp_percentage,
                salience: Some(resp.salience.total),
                cycle: Some(resp.cycle),
            })),
            Err(e) => Err((StatusCode::BAD_GATEWAY, e)),
        }
    }
}

async fn stream_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StreamRequest>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

    let ollama = state.ollama.clone();
    let prompt = req.message;
    let system = req.system;

    tokio::spawn(async move {
        let _ = ollama.generate_stream(&prompt, system.as_deref(), tx).await;
    });

    let stream = async_stream::stream! {
        while let Some(token) = rx.recv().await {
            let data = serde_json::json!({ "token": token });
            yield Ok(Event::default().data(data.to_string()));
        }
        yield Ok(Event::default().data(r#"{"done":true}"#.to_string()));
    };

    Sse::new(stream)
}

async fn peers(State(state): State<Arc<AppState>>) -> Json<PeersResponse> {
    let (self_machine, peer_list, version) = match &state.fleet {
        Some(fleet) => {
            let peers: Vec<PeerInfo> = fleet.get_peers()
                .into_iter()
                .map(|(name, info)| PeerInfo {
                    name: name.to_string(),
                    gateway_url: format!("http://{}:{}", info.gateway_host, info.gateway_port),
                    pool: info.pool.clone(),
                    model: info.model_default.clone(),
                    device: info.device.clone(),
                    hardware: info.hardware.clone(),
                })
                .collect();
            ("sprout".to_string(), peers, fleet.version())
        }
        None => ("sprout".to_string(), vec![], 0),
    };

    Json(PeersResponse {
        self_machine,
        peers: peer_list,
        fleet_version: version,
    })
}

fn run_simulation(cycles: u64) {
    println!("sage-daemon --simulate {cycles}");
    println!("{:-<60}", "");

    let mut ctrl = MetabolicController::with_defaults();
    let mut surprise = SurpriseDetector::with_defaults();
    let mut state_counts = std::collections::HashMap::<MetabolicState, u32>::new();
    let start = Instant::now();

    for i in 0..cycles {
        let salience: f64 = if (i % 80) < 15 { 0.6 } else { 0.15 };
        let obs = if (i % 80) < 15 { 10.0 + (i as f64 * 0.1) } else { 1.0 };
        let s = surprise.compute(obs, "sim");

        let data = CycleData {
            max_salience: salience.max(s),
            ..Default::default()
        };
        let state = ctrl.update(&data);
        *state_counts.entry(state).or_insert(0) += 1;

        if i < 20 || i % (cycles / 10).max(1) == 0 {
            println!(
                "cycle {:>5}  state={:<7}  ATP={:5.1}  salience={:.2}  surprise={:.3}",
                i, state.as_str(), ctrl.atp_current, salience, s
            );
        }
    }

    let elapsed = start.elapsed();
    println!("{:-<60}", "");
    println!("{cycles} cycles in {:.3}s ({:.0} cycles/s)", elapsed.as_secs_f64(), cycles as f64 / elapsed.as_secs_f64());
    println!("transitions: {}", ctrl.history.len());
    println!("state distribution:");
    let mut sorted: Vec<_> = state_counts.iter().collect();
    sorted.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (state, count) in sorted {
        println!("  {:<10} {:>5} ({:.1}%)", state.as_str(), count, *count as f64 / cycles as f64 * 100.0);
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "--simulate" {
        let cycles = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
        run_simulation(cycles);
        return;
    }

    tracing_subscriber::fmt::init();

    let model = std::env::var("SAGE_MODEL").unwrap_or_else(|_| "qwen3.5:0.8b".to_string());

    let fleet = FleetRegistry::load("sprout", std::path::Path::new(FLEET_JSON)).ok();
    if let Some(ref f) = fleet {
        info!("fleet loaded: {} machines, v{}", f.fleet_size(), f.version());
    }

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let (msg_tx, msg_rx) = tokio::sync::mpsc::channel(64);
    let consciousness_handle = ConsciousnessHandle::new(msg_tx);

    let experience = ExperienceBuffer::with_defaults(std::path::Path::new(EXPERIENCE_PATH));
    info!("experience buffer: {} existing entries", experience.count());

    let consciousness_loop = ConsciousnessLoop::new(
        OllamaClient::default_local(&model),
        experience,
        msg_rx,
        "sprout",
        &model,
    );

    let loop_shutdown = shutdown_rx.clone();
    let loop_handle = tokio::spawn(async move {
        consciousness_loop.run(loop_shutdown).await;
    });

    let state = Arc::new(AppState {
        surprise: Mutex::new(SurpriseDetector::with_defaults()),
        novelty: Mutex::new(NoveltyDetector::with_defaults()),
        arousal: Mutex::new(ArousalDetector::with_defaults()),
        reward: Mutex::new(RewardEstimator::with_defaults()),
        conflict: Mutex::new(ConflictDetector::with_defaults()),
        metabolic: Mutex::new(MetabolicController::with_defaults()),
        consciousness: consciousness_handle,
        ollama: OllamaClient::default_local(&model),
        fleet,
        started: Instant::now(),
        model: model.clone(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/snarc/observe", post(snarc_observe))
        .route("/metabolic/cycle", post(metabolic_cycle))
        .route("/chat", post(chat))
        .route("/stream", post(stream_chat))
        .route("/peers", get(peers))
        .with_state(state);

    let addr = format!("0.0.0.0:{PORT}");
    info!("sage-daemon listening on {addr} (model={model}, consciousness=active)");
    println!("sage-daemon listening on {addr} (model={model}, consciousness=active)");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            let mut sigterm = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::terminate()
            ).expect("failed to install SIGTERM handler");

            tokio::select! {
                _ = ctrl_c => info!("received Ctrl+C"),
                _ = sigterm.recv() => info!("received SIGTERM"),
            }

            info!("initiating graceful shutdown...");
            let _ = shutdown_tx.send(true);
        });

    server.await.unwrap();

    let _ = loop_handle.await;
    info!("sage-daemon shut down cleanly");
}
