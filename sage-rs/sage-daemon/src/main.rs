use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::info;

use sage_lib::metabolic::controller::{CycleData, MetabolicController, MetabolicState};
use sage_lib::snarc::arousal::ArousalDetector;
use sage_lib::snarc::conflict::ConflictDetector;
use sage_lib::snarc::novelty::NoveltyDetector;
use sage_lib::snarc::reward::RewardEstimator;
use sage_lib::snarc::surprise::SurpriseDetector;
use sage_lib::snarc::temporal;

struct AppState {
    surprise: Mutex<SurpriseDetector>,
    novelty: Mutex<NoveltyDetector>,
    arousal: Mutex<ArousalDetector>,
    reward: Mutex<RewardEstimator>,
    conflict: Mutex<ConflictDetector>,
    metabolic: Mutex<MetabolicController>,
    started: Instant,
}

// --- Request/Response types ---

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    uptime_secs: f64,
    version: &'static str,
    port: u16,
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

const PORT: u16 = 8760;

// --- Handlers ---

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        uptime_secs: state.started.elapsed().as_secs_f64(),
        version: env!("CARGO_PKG_VERSION"),
        port: PORT,
    })
}

async fn status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    let ctrl = state.metabolic.lock().await;
    Json(StatusResponse {
        daemon: "sage-daemon",
        sprint: "1 — full SNARC + metabolic",
        snarc_detectors: vec!["surprise", "novelty", "arousal", "reward", "conflict"],
        half_lives: temporal::DEFAULT_HALF_LIVES.to_vec(),
        metabolic_state: ctrl.current_state.as_str(),
        atp_percentage: ctrl.atp_percentage(),
        total_cycles: ctrl.total_cycles,
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

    Json(SnarcResponse {
        surprise,
        novelty,
        arousal,
        reward,
        sensor_id,
    })
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

fn run_simulation(cycles: u64) {
    use sage_lib::metabolic::controller::MetabolicController;

    println!("sage-daemon --simulate {cycles}");
    println!("{:-<60}", "");

    let mut ctrl = MetabolicController::with_defaults();
    let mut surprise = SurpriseDetector::with_defaults();
    let mut state_counts = std::collections::HashMap::<MetabolicState, u32>::new();
    let start = Instant::now();

    for i in 0..cycles {
        // Synthetic signal: periodic bursts of salience
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
                i,
                state.as_str(),
                ctrl.atp_current,
                salience,
                s
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
    // Check for --simulate flag
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "--simulate" {
        let cycles = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
        run_simulation(cycles);
        return;
    }

    tracing_subscriber::fmt::init();

    let state = Arc::new(AppState {
        surprise: Mutex::new(SurpriseDetector::with_defaults()),
        novelty: Mutex::new(NoveltyDetector::with_defaults()),
        arousal: Mutex::new(ArousalDetector::with_defaults()),
        reward: Mutex::new(RewardEstimator::with_defaults()),
        conflict: Mutex::new(ConflictDetector::with_defaults()),
        metabolic: Mutex::new(MetabolicController::with_defaults()),
        started: Instant::now(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/snarc/observe", post(snarc_observe))
        .route("/metabolic/cycle", post(metabolic_cycle))
        .with_state(state);

    let addr = format!("0.0.0.0:{PORT}");
    info!("sage-daemon listening on {addr}");
    println!("sage-daemon listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
