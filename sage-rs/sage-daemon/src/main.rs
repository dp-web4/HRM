use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::info;

use sage_lib::snarc::surprise::SurpriseDetector;
use sage_lib::snarc::temporal;

struct AppState {
    detector: Mutex<SurpriseDetector>,
    started: Instant,
}

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
}

#[derive(Deserialize)]
struct SurpriseRequest {
    observation: f64,
    sensor_id: Option<String>,
}

#[derive(Serialize)]
struct SurpriseResponse {
    surprise: f64,
    sensor_id: String,
}

const PORT: u16 = 8760;

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        uptime_secs: state.started.elapsed().as_secs_f64(),
        version: env!("CARGO_PKG_VERSION"),
        port: PORT,
    })
}

async fn status() -> Json<StatusResponse> {
    Json(StatusResponse {
        daemon: "sage-daemon",
        sprint: "0 — SNARC math foundation",
        snarc_detectors: vec!["surprise"],
        half_lives: temporal::DEFAULT_HALF_LIVES.to_vec(),
    })
}

async fn surprise(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SurpriseRequest>,
) -> Result<Json<SurpriseResponse>, StatusCode> {
    let sensor_id = req.sensor_id.unwrap_or_else(|| "default".to_string());
    let mut detector = state.detector.lock().await;
    let score = detector.compute(req.observation, &sensor_id);
    Ok(Json(SurpriseResponse {
        surprise: score,
        sensor_id,
    }))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = Arc::new(AppState {
        detector: Mutex::new(SurpriseDetector::with_defaults()),
        started: Instant::now(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/snarc/surprise", post(surprise))
        .with_state(state);

    let addr = format!("0.0.0.0:{PORT}");
    info!("sage-daemon listening on {addr}");
    println!("sage-daemon listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
