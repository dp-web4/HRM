use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct OllamaClient {
    base_url: String,
    model: String,
    client: Client,
}

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    // Thinking models (qwen3.5, ...) otherwise default to a <think> trace that consumes the token
    // budget and returns an empty visible answer (the request hung to the 120s timeout, blank text).
    // Cognition wants a direct reply, so thinking is off; num_predict bounds the length.
    think: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    options: GenOpts,
}

#[derive(Serialize)]
struct GenOpts {
    num_predict: i32,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
    #[serde(default)]
    done: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Option<ChatMessageFragment>,
    #[serde(default)]
    done: bool,
}

#[derive(Deserialize)]
struct ChatMessageFragment {
    content: String,
}

#[derive(Deserialize)]
struct ModelListResponse {
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap(),
        }
    }

    pub fn default_local(model: &str) -> Self {
        Self::new("http://localhost:11434", model)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Non-streaming generate. Returns the full response text.
    pub async fn generate(&self, prompt: &str, system: Option<&str>) -> Result<String, String> {
        let req = GenerateRequest {
            model: &self.model,
            prompt,
            stream: false,
            think: false,
            system,
            options: GenOpts { num_predict: 300 },
        };

        let resp = self.client
            .post(format!("{}/api/generate", self.base_url))
            .json(&req)
            .send()
            .await
            .map_err(|e| format!("ollama request failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("ollama returned {}", resp.status()));
        }

        let body: GenerateResponse = resp.json().await
            .map_err(|e| format!("ollama response parse error: {e}"))?;

        Ok(body.response)
    }

    /// Non-streaming chat with message history.
    pub async fn chat(&self, messages: &[ChatMessage]) -> Result<String, String> {
        let req = ChatRequest {
            model: &self.model,
            messages,
            stream: false,
        };

        let resp = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&req)
            .send()
            .await
            .map_err(|e| format!("ollama chat failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("ollama returned {}", resp.status()));
        }

        let body: ChatResponse = resp.json().await
            .map_err(|e| format!("ollama chat parse error: {e}"))?;

        Ok(body.message.map(|m| m.content).unwrap_or_default())
    }

    /// Streaming generate — sends tokens to the channel as they arrive.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        system: Option<&str>,
        tx: mpsc::Sender<String>,
    ) -> Result<(), String> {
        let req = GenerateRequest {
            model: &self.model,
            prompt,
            stream: true,
            think: false,
            system,
            options: GenOpts { num_predict: 512 },
        };

        let resp = self.client
            .post(format!("{}/api/generate", self.base_url))
            .json(&req)
            .send()
            .await
            .map_err(|e| format!("ollama stream request failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("ollama returned {}", resp.status()));
        }

        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| format!("stream read error: {e}"))?;
            // Each line is a JSON object
            for line in bytes.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                if let Ok(parsed) = serde_json::from_slice::<GenerateResponse>(line) {
                    if !parsed.response.is_empty() {
                        let _ = tx.send(parsed.response).await;
                    }
                    if parsed.done {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    /// Streaming chat — sends tokens to the channel as they arrive.
    pub async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tx: mpsc::Sender<String>,
    ) -> Result<(), String> {
        let req = ChatRequest {
            model: &self.model,
            messages,
            stream: true,
        };

        let resp = self.client
            .post(format!("{}/api/chat", self.base_url))
            .json(&req)
            .send()
            .await
            .map_err(|e| format!("ollama chat stream failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("ollama returned {}", resp.status()));
        }

        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.map_err(|e| format!("stream read error: {e}"))?;
            for line in bytes.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                if let Ok(parsed) = serde_json::from_slice::<ChatResponse>(line) {
                    if let Some(msg) = parsed.message {
                        if !msg.content.is_empty() {
                            let _ = tx.send(msg.content).await;
                        }
                    }
                    if parsed.done {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, String> {
        let resp = self.client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map_err(|e| format!("ollama list models failed: {e}"))?;

        let body: ModelListResponse = resp.json().await
            .map_err(|e| format!("ollama model list parse error: {e}"))?;

        Ok(body.models)
    }

    /// Check if Ollama is reachable.
    pub async fn is_available(&self) -> bool {
        self.client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .is_ok()
    }
}
