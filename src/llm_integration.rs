use anyhow::{Context, Result};
use clap::ValueEnum;
use log::{debug, error, info};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Duration;

/// Type alias for token generation callback
pub type TokenCallback = Box<dyn FnMut(String, bool) -> bool>;

/// Enum representing the LLM backend to use
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LlmBackend {
    /// Use llama.c via FFI
    LlamaC,
    /// Use llama.rs (pure Rust implementation)
    LlamaRs,
    /// Use Ollama API
    Ollama,
    /// Use LM Studio API
    LmStudio,
    /// Use llama-server (OpenAI-compatible local server)
    LlamaServer,
}

/// Statistics returned from generation, populated from backend-native metadata
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Time spent evaluating the prompt (nanoseconds)
    pub prompt_eval_duration_ns: Option<u64>,
    /// Time spent generating tokens (nanoseconds)
    pub eval_duration_ns: Option<u64>,
    /// Number of tokens in the prompt
    pub prompt_eval_count: Option<i32>,
    /// Number of tokens generated
    pub eval_count: Option<i32>,
    /// Total generation duration (nanoseconds)
    pub total_duration_ns: Option<u64>,
    /// Model load duration (nanoseconds)
    pub load_duration_ns: Option<u64>,
    /// Tokens per second as reported by backend
    pub backend_tokens_per_second: Option<f64>,
}

/// Trait for LLM models
pub trait LlmModel {
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        options: &GenerationOptions,
        callback: TokenCallback,
    ) -> Result<GenerationStats>;
    fn get_model_info(&mut self) -> Result<Option<ModelInfo>> {
        Ok(None)
    }
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model family/architecture
    pub family: Option<String>,
    /// Parameter count (in billions)
    pub parameter_count: Option<f64>,
    /// Quantization level
    pub quantization: Option<String>,
    /// Context length
    pub context_length: Option<usize>,
    /// Additional model metadata
    pub metadata: serde_json::Value,
}

/// Generation options for LLMs
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub context_length: Option<usize>,
    pub mirostat: Option<i32>,
}

/// Create an LLM model based on the specified backend
pub fn create_llm_model(model_path: &Path, backend: LlmBackend) -> Result<Box<dyn LlmModel>> {
    match backend {
        LlmBackend::LlamaC => {
            info!("Creating llama.c model from {}", model_path.display());
            Ok(Box::new(LlamaCModel::new(model_path)?))
        }
        LlmBackend::LlamaRs => {
            info!("Creating llama.rs model from {}", model_path.display());
            Ok(Box::new(LlamaRsModel::new(model_path)?))
        }
        LlmBackend::Ollama => {
            info!("Creating Ollama model from {}", model_path.display());
            Ok(Box::new(OllamaModel::new(model_path)?))
        }
        LlmBackend::LmStudio => {
            info!("Creating LM Studio model from {}", model_path.display());
            Ok(Box::new(LmStudioModel::new(model_path)?))
        }
        LlmBackend::LlamaServer => {
            info!("Creating llama-server model from {}", model_path.display());
            Ok(Box::new(LlamaServerModel::new(model_path)?))
        }
    }
}

// ============================================================================
// llama.c Implementation (simulated)
// ============================================================================

/// Implementation for llama.c
pub struct LlamaCModel {
    #[allow(dead_code)]
    model_path: String,
}

impl LlamaCModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        debug!("Loading llama.c model from {}", model_path.display());
        Ok(LlamaCModel {
            model_path: model_path.display().to_string(),
        })
    }
}

impl LlmModel for LlamaCModel {
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        _options: &GenerationOptions,
        mut callback: TokenCallback,
    ) -> Result<GenerationStats> {
        debug!(
            "Generating with llama.c model, prompt: {}, max_tokens: {}",
            prompt, max_tokens
        );

        // Simulated token generation
        for i in 0..max_tokens.min(10) {
            let token = format!("token_{}", i);
            let is_first = i == 0;

            if !callback(token, is_first) {
                break;
            }
        }

        Ok(GenerationStats::default())
    }
}

// ============================================================================
// llama.rs Implementation (simulated)
// ============================================================================

/// Implementation for llama.rs
pub struct LlamaRsModel {
    #[allow(dead_code)]
    model_path: String,
}

impl LlamaRsModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        debug!("Loading llama.rs model from {}", model_path.display());
        Ok(LlamaRsModel {
            model_path: model_path.display().to_string(),
        })
    }
}

impl LlmModel for LlamaRsModel {
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        _options: &GenerationOptions,
        mut callback: TokenCallback,
    ) -> Result<GenerationStats> {
        debug!(
            "Generating with llama.rs model, prompt: {}, max_tokens: {}",
            prompt, max_tokens
        );

        // Simulated token generation
        for i in 0..max_tokens.min(10) {
            let token = format!("token_{}", i);
            let is_first = i == 0;

            if !callback(token, is_first) {
                break;
            }
        }

        Ok(GenerationStats::default())
    }
}

// ============================================================================
// Ollama Implementation
// ============================================================================

/// Request structure for Ollama API
#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: Option<OllamaOptions>,
}

/// Options for Ollama API
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaOptions {
    pub num_keep: Option<i32>,
    pub seed: Option<i32>,
    pub num_predict: Option<i32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub temperature: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub mirostat: Option<i32>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub penalize_newline: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub numa: Option<bool>,
    pub num_ctx: Option<i32>,
    pub num_batch: Option<i32>,
    pub num_gpu: Option<i32>,
    pub main_gpu: Option<i32>,
    pub low_vram: Option<bool>,
    pub vocab_only: Option<bool>,
    pub use_mmap: Option<bool>,
    pub use_mlock: Option<bool>,
    pub num_thread: Option<i32>,
}

/// Response structure from Ollama API
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct OllamaResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub context: Vec<i32>,
    #[serde(default)]
    pub total_duration: u64,
    #[serde(default)]
    pub load_duration: u64,
    #[serde(default)]
    pub prompt_eval_count: i32,
    #[serde(default)]
    pub prompt_eval_duration: u64,
    #[serde(default)]
    pub eval_count: i32,
    #[serde(default)]
    pub eval_duration: u64,
}

/// Ollama model info response
#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
struct OllamaModelInfoResponse {
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: OllamaModelDetails,
}

/// Ollama model details
#[derive(Deserialize, Debug, Serialize, Clone)]
struct OllamaModelDetails {
    format: String,
    family: String,
    families: Option<Vec<String>>,
    parameter_size: String,
    quantization_level: Option<String>,
    context_length: usize,
}

/// Implementation for Ollama API
pub struct OllamaModel {
    model_name: String,
    client: Client,
    model_info: Option<ModelInfo>,
    base_url: String,
}

impl OllamaModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        Self::new_with_base_url(model_path, "http://localhost:11434")
    }

    pub fn new_with_base_url(model_path: &Path, base_url: &str) -> Result<Self> {
        let model_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        debug!("Using Ollama model: {}", model_name);

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("Failed to create HTTP client")?;

        let mut model = OllamaModel {
            model_name,
            client,
            model_info: None,
            base_url: base_url.to_string(),
        };

        if let Ok(Some(info)) = model.get_model_info() {
            model.model_info = Some(info);
        }

        Ok(model)
    }

    fn fetch_model_info(&self) -> Result<OllamaModelInfoResponse> {
        let url = format!("{}/api/show", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "name": self.model_name }))
            .send()
            .context("Failed to send request to Ollama API for model info")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!("Ollama API error: {} - {}", status, error_text);
            return Err(anyhow::anyhow!(
                "Ollama API error: {} - {}",
                status,
                error_text
            ));
        }

        response
            .json()
            .context("Failed to parse Ollama model info response")
    }

    fn build_options(max_tokens: usize, options: &GenerationOptions) -> OllamaOptions {
        OllamaOptions {
            num_predict: i32::try_from(max_tokens).ok(),
            temperature: options.temperature,
            top_k: options.top_k,
            top_p: options.top_p,
            repeat_penalty: options.repeat_penalty,
            mirostat: options.mirostat,
            num_ctx: options.context_length.and_then(|c| i32::try_from(c).ok()),
            ..Default::default()
        }
    }
}

impl LlmModel for OllamaModel {
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        options: &GenerationOptions,
        mut callback: TokenCallback,
    ) -> Result<GenerationStats> {
        debug!(
            "Generating with Ollama model: {}, prompt: {}, max_tokens: {}",
            self.model_name, prompt, max_tokens
        );

        let request = OllamaRequest {
            model: self.model_name.clone(),
            prompt: prompt.to_string(),
            stream: true,
            options: Some(Self::build_options(max_tokens, options)),
        };

        let url = format!("{}/api/generate", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to send request to Ollama API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!("Ollama API error: {} - {}", status, error_text);
            return Err(anyhow::anyhow!(
                "Ollama API error: {} - {}",
                status,
                error_text
            ));
        }

        // Incremental streaming with BufReader
        let reader = BufReader::new(response);
        let mut is_first = true;
        let mut final_response: Option<OllamaResponse> = None;

        for line_result in reader.lines() {
            let line = line_result.context("Failed to read line from Ollama stream")?;
            if line.is_empty() {
                continue;
            }

            let ollama_response: OllamaResponse =
                serde_json::from_str(&line).context("Failed to parse Ollama API response")?;

            if !ollama_response.response.is_empty() {
                if !callback(ollama_response.response.clone(), is_first) {
                    break;
                }
                is_first = false;
            }

            if ollama_response.done {
                final_response = Some(ollama_response);
                break;
            }
        }

        // Extract stats from final response
        let mut stats = GenerationStats::default();
        if let Some(final_resp) = final_response {
            stats.prompt_eval_duration_ns = if final_resp.prompt_eval_duration > 0 {
                Some(final_resp.prompt_eval_duration)
            } else {
                None
            };
            stats.eval_duration_ns = if final_resp.eval_duration > 0 {
                Some(final_resp.eval_duration)
            } else {
                None
            };
            stats.prompt_eval_count = if final_resp.prompt_eval_count > 0 {
                Some(final_resp.prompt_eval_count)
            } else {
                None
            };
            stats.eval_count = if final_resp.eval_count > 0 {
                Some(final_resp.eval_count)
            } else {
                None
            };
            stats.total_duration_ns = if final_resp.total_duration > 0 {
                Some(final_resp.total_duration)
            } else {
                None
            };
            stats.load_duration_ns = if final_resp.load_duration > 0 {
                Some(final_resp.load_duration)
            } else {
                None
            };

            // Compute backend tokens per second if we have eval duration
            if final_resp.eval_duration > 0 && final_resp.eval_count > 0 {
                let eval_seconds = final_resp.eval_duration as f64 / 1_000_000_000.0;
                stats.backend_tokens_per_second = Some(final_resp.eval_count as f64 / eval_seconds);
            }
        }

        Ok(stats)
    }

    fn get_model_info(&mut self) -> Result<Option<ModelInfo>> {
        if let Some(ref info) = self.model_info {
            return Ok(Some(info.clone()));
        }

        match self.fetch_model_info() {
            Ok(ollama_info) => {
                let parameter_count = ollama_info
                    .details
                    .parameter_size
                    .trim_end_matches(|c: char| !c.is_numeric())
                    .parse::<f64>()
                    .ok();

                let details_json =
                    serde_json::to_value(ollama_info.details.clone()).unwrap_or_default();

                let model_info = ModelInfo {
                    name: ollama_info.model.clone(),
                    family: Some(ollama_info.details.family.clone()),
                    parameter_count,
                    quantization: ollama_info.details.quantization_level.clone(),
                    context_length: Some(ollama_info.details.context_length),
                    metadata: details_json,
                };

                self.model_info = Some(model_info.clone());
                Ok(Some(model_info))
            }
            Err(e) => {
                debug!("Failed to fetch model info: {}", e);
                Ok(None)
            }
        }
    }
}

// ============================================================================
// LM Studio Implementation
// ============================================================================

/// Response structure from LM Studio API (OpenAI compatible)
#[derive(Debug, Clone, Deserialize)]
struct LmStudioStreamResponse {
    choices: Vec<LmStudioStreamChoice>,
    #[serde(default)]
    usage: Option<LmStudioUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct LmStudioStreamChoice {
    delta: LmStudioStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct LmStudioStreamDelta {
    content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct LmStudioUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    #[serde(rename = "total_tokens")]
    _total_tokens: i32,
}

/// LM Studio model info response
#[derive(Deserialize, Debug, Clone, Serialize)]
struct LmStudioModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Deserialize, Debug, Clone)]
struct LmStudioModelList {
    data: Vec<LmStudioModelInfo>,
}

/// Implementation for LM Studio API
pub struct LmStudioModel {
    model_name: String,
    client: Client,
    model_info: Option<ModelInfo>,
    base_url: String,
}

impl LmStudioModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        Self::new_with_base_url(model_path, "http://localhost:1234/v1")
    }

    pub fn new_with_base_url(model_path: &Path, base_url: &str) -> Result<Self> {
        let model_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        debug!("Using LM Studio model: {}", model_name);

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("Failed to create HTTP client")?;

        let mut model = LmStudioModel {
            model_name,
            client,
            model_info: None,
            base_url: base_url.to_string(),
        };

        if let Ok(Some(info)) = model.get_model_info() {
            model.model_info = Some(info);
        }

        Ok(model)
    }

    fn fetch_model_info(&self) -> Result<LmStudioModelInfo> {
        let url = format!("{}/models", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .context("Failed to send request to LM Studio API for model info")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!("LM Studio API error: {} - {}", status, error_text);
            return Err(anyhow::anyhow!(
                "LM Studio API error: {} - {}",
                status,
                error_text
            ));
        }

        let model_list: LmStudioModelList = response
            .json()
            .context("Failed to parse LM Studio model list response")?;

        model_list
            .data
            .into_iter()
            .find(|m| m.id == self.model_name)
            .context(format!(
                "Model '{}' not found in LM Studio",
                self.model_name
            ))
    }
}

impl LlmModel for LmStudioModel {
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        options: &GenerationOptions,
        mut callback: TokenCallback,
    ) -> Result<GenerationStats> {
        debug!(
            "Generating with LM Studio model: {}, prompt: {}, max_tokens: {}",
            self.model_name, prompt, max_tokens
        );

        // Build request with options
        let mut request_body = serde_json::json!({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": true,
            "stream_options": {
                "include_usage": true
            }
        });

        // Add sampling options if provided
        if let Some(temp) = options.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = options.top_p {
            request_body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(top_k) = options.top_k {
            request_body["top_k"] = serde_json::json!(top_k);
        }
        if let Some(repeat_penalty) = options.repeat_penalty {
            request_body["repeat_penalty"] = serde_json::json!(repeat_penalty);
        }
        if let Some(mirostat) = options.mirostat {
            request_body["mirostat"] = serde_json::json!(mirostat);
        }

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(url)
            .json(&request_body)
            .send()
            .context("Failed to send request to LM Studio API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!("LM Studio API error: {} - {}", status, error_text);
            return Err(anyhow::anyhow!(
                "LM Studio API error: {} - {}",
                status,
                error_text
            ));
        }

        // Incremental SSE parsing with BufReader
        let reader = BufReader::new(response);
        let mut is_first = true;
        let mut stats = GenerationStats::default();

        for line_result in reader.lines() {
            let line = line_result.context("Failed to read line from LM Studio stream")?;

            if !line.starts_with("data: ") {
                continue;
            }

            let json_str = &line[6..];
            if json_str == "[DONE]" {
                break;
            }

            let lm_studio_response: LmStudioStreamResponse = match serde_json::from_str(json_str) {
                Ok(res) => res,
                Err(e) => {
                    debug!("Failed to parse LM Studio API response chunk: {}", e);
                    continue;
                }
            };

            // Capture usage stats if present
            if let Some(usage) = &lm_studio_response.usage {
                stats.prompt_eval_count = Some(usage.prompt_tokens);
                stats.eval_count = Some(usage.completion_tokens);
            }

            if let Some(choice) = lm_studio_response.choices.get(0) {
                if let Some(content) = &choice.delta.content {
                    if !callback(content.clone(), is_first) {
                        break;
                    }
                    is_first = false;
                }
                if choice.finish_reason.is_some() {
                    break;
                }
            }
        }

        Ok(stats)
    }

    fn get_model_info(&mut self) -> Result<Option<ModelInfo>> {
        if let Some(ref info) = self.model_info {
            return Ok(Some(info.clone()));
        }

        match self.fetch_model_info() {
            Ok(lm_studio_info) => {
                let model_info = ModelInfo {
                    name: lm_studio_info.id.clone(),
                    family: None,
                    parameter_count: None,
                    quantization: None,
                    context_length: None,
                    metadata: serde_json::to_value(lm_studio_info).unwrap_or_default(),
                };

                self.model_info = Some(model_info.clone());
                Ok(Some(model_info))
            }
            Err(e) => {
                debug!("Failed to fetch model info from LM Studio: {}", e);
                Ok(None)
            }
        }
    }
}

// ============================================================================
// llama-server Implementation
// ============================================================================

/// Response structure from llama-server (OpenAI-compatible with timing extensions)
#[derive(Debug, Clone, Deserialize)]
struct LlamaServerStreamResponse {
    choices: Vec<LlamaServerStreamChoice>,
    #[serde(default)]
    usage: Option<LlamaServerUsage>,
    #[serde(default)]
    timings: Option<LlamaServerTimings>,
}

#[derive(Debug, Clone, Deserialize)]
struct LlamaServerStreamChoice {
    delta: LlamaServerStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct LlamaServerStreamDelta {
    content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct LlamaServerUsage {
    prompt_tokens: i32,
    completion_tokens: i32,
    #[serde(rename = "total_tokens")]
    _total_tokens: i32,
}

/// Timing information from llama-server (in milliseconds)
#[derive(Debug, Clone, Deserialize)]
struct LlamaServerTimings {
    #[serde(default)]
    prompt_n: Option<i32>,
    #[serde(default)]
    prompt_ms: Option<f64>,
    #[serde(default)]
    predicted_n: Option<i32>,
    #[serde(default)]
    predicted_ms: Option<f64>,
}

/// Implementation for llama-server API
pub struct LlamaServerModel {
    model_name: String,
    client: Client,
    model_info: Option<ModelInfo>,
    base_url: String,
}

impl LlamaServerModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        Self::new_with_base_url(model_path, "http://localhost:8080/v1")
    }

    pub fn new_with_base_url(model_path: &Path, base_url: &str) -> Result<Self> {
        let model_name = path_to_model_name(model_path);

        debug!("Using llama-server model: {}", model_name);

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("Failed to create HTTP client")?;

        let mut model = LlamaServerModel {
            model_name,
            client,
            model_info: None,
            base_url: base_url.to_string(),
        };

        if let Ok(Some(info)) = model.get_model_info() {
            model.model_info = Some(info);
        }

        Ok(model)
    }
}

/// Extract model name from path
fn path_to_model_name(model_path: &Path) -> String {
    model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string()
}

impl LlmModel for LlamaServerModel {
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        options: &GenerationOptions,
        mut callback: TokenCallback,
    ) -> Result<GenerationStats> {
        debug!(
            "Generating with llama-server model: {}, prompt: {}, max_tokens: {}",
            self.model_name, prompt, max_tokens
        );

        // Build request with options
        let mut request_body = serde_json::json!({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": true,
            "stream_options": {
                "include_usage": true
            }
        });

        // Add sampling options if provided
        if let Some(temp) = options.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = options.top_p {
            request_body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(top_k) = options.top_k {
            request_body["top_k"] = serde_json::json!(top_k);
        }
        if let Some(repeat_penalty) = options.repeat_penalty {
            request_body["repeat_penalty"] = serde_json::json!(repeat_penalty);
        }
        if let Some(mirostat) = options.mirostat {
            request_body["mirostat"] = serde_json::json!(mirostat);
        }

        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(url)
            .json(&request_body)
            .send()
            .context("Failed to send request to llama-server API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!("llama-server API error: {} - {}", status, error_text);
            return Err(anyhow::anyhow!(
                "llama-server API error: {} - {}",
                status,
                error_text
            ));
        }

        // Incremental SSE parsing with BufReader
        let reader = BufReader::new(response);
        let mut is_first = true;
        let mut stats = GenerationStats::default();

        for line_result in reader.lines() {
            let line = line_result.context("Failed to read line from llama-server stream")?;

            if !line.starts_with("data: ") {
                continue;
            }

            let json_str = &line[6..];
            if json_str == "[DONE]" {
                break;
            }

            let stream_response: LlamaServerStreamResponse = match serde_json::from_str(json_str) {
                Ok(res) => res,
                Err(e) => {
                    debug!("Failed to parse llama-server response chunk: {}", e);
                    continue;
                }
            };

            // Capture usage stats if present
            if let Some(usage) = &stream_response.usage {
                stats.prompt_eval_count = Some(usage.prompt_tokens);
                stats.eval_count = Some(usage.completion_tokens);
            }

            // Capture timing info if present (convert ms to ns)
            if let Some(timings) = &stream_response.timings {
                if let Some(prompt_ms) = timings.prompt_ms {
                    stats.prompt_eval_duration_ns = Some((prompt_ms * 1_000_000.0) as u64);
                }
                if let Some(predicted_ms) = timings.predicted_ms {
                    stats.eval_duration_ns = Some((predicted_ms * 1_000_000.0) as u64);
                }
                if let Some(prompt_n) = timings.prompt_n {
                    stats.prompt_eval_count = Some(prompt_n);
                }
                if let Some(predicted_n) = timings.predicted_n {
                    stats.eval_count = Some(predicted_n);
                }

                // Calculate backend tokens per second
                if let (Some(predicted_ms), Some(predicted_n)) =
                    (timings.predicted_ms, timings.predicted_n)
                {
                    if predicted_ms > 0.0 && predicted_n > 0 {
                        let seconds = predicted_ms / 1000.0;
                        stats.backend_tokens_per_second = Some(predicted_n as f64 / seconds);
                    }
                }
            }

            if let Some(choice) = stream_response.choices.get(0) {
                if let Some(content) = &choice.delta.content {
                    if !callback(content.clone(), is_first) {
                        break;
                    }
                    is_first = false;
                }
                if choice.finish_reason.is_some() {
                    break;
                }
            }
        }

        Ok(stats)
    }

    fn get_model_info(&mut self) -> Result<Option<ModelInfo>> {
        // llama-server doesn't have a reliable model info endpoint
        // Return basic info from the model name
        Ok(Some(ModelInfo {
            name: self.model_name.clone(),
            family: None,
            parameter_count: None,
            quantization: None,
            context_length: None,
            metadata: serde_json::json!({}),
        }))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use std::cell::RefCell;
    use std::path::PathBuf;
    use std::rc::Rc;

    #[test]
    fn test_lm_studio_get_model_info() {
        let mut server = Server::new();
        let model_name = "test-model";
        let base_url = server.url();

        let _m = server
            .mock("GET", "/models")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"data": [{{"id": "{}","object": "model","created": 1234567890,"owned_by": "test-owner"}}]}}"#,
                model_name
            ))
            .create();

        let model_path = PathBuf::from(model_name);
        let mut model = LmStudioModel::new_with_base_url(&model_path, &base_url).unwrap();

        let model_info = model.get_model_info().unwrap().unwrap();

        assert_eq!(model_info.name, model_name);
        assert!(model_info.metadata.get("owned_by").is_some());
    }

    #[test]
    fn test_lm_studio_generate() {
        let mut server = Server::new();
        let model_name = "test-model";
        let base_url = server.url();

        let stream_body = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\ndata: [DONE]\n\n";

        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(stream_body)
            .create();

        let model_path = PathBuf::from(model_name);
        let mut model = LmStudioModel::new_with_base_url(&model_path, &base_url).unwrap();

        let tokens = Rc::new(RefCell::new(Vec::new()));
        let tokens_clone = tokens.clone();

        let callback = Box::new(move |token: String, _is_first: bool| {
            tokens_clone.borrow_mut().push(token);
            true
        });

        let options = GenerationOptions::default();
        model
            .generate("test prompt", 10, &options, callback)
            .unwrap();

        assert_eq!(*tokens.borrow(), vec!["hello", " world"]);
    }

    #[test]
    fn test_llama_server_generate() {
        let mut server = Server::new();
        let model_name = "test-model";
        let base_url = server.url();

        // Include usage and timings in the final chunk
        let stream_body = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7},\"timings\":{\"prompt_n\":5,\"prompt_ms\":10.5,\"predicted_n\":2,\"predicted_ms\":50.0}}\n\ndata: [DONE]\n\n";

        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(stream_body)
            .create();

        let model_path = PathBuf::from(model_name);
        let mut model = LlamaServerModel::new_with_base_url(&model_path, &base_url).unwrap();

        let tokens = Rc::new(RefCell::new(Vec::new()));
        let tokens_clone = tokens.clone();

        let callback = Box::new(move |token: String, _is_first: bool| {
            tokens_clone.borrow_mut().push(token);
            true
        });

        let options = GenerationOptions::default();
        let stats = model
            .generate("test prompt", 10, &options, callback)
            .unwrap();

        assert_eq!(*tokens.borrow(), vec!["hello", " world"]);

        // Verify stats were captured
        assert_eq!(stats.prompt_eval_count, Some(5));
        assert_eq!(stats.eval_count, Some(2));
        assert!(stats.prompt_eval_duration_ns.is_some());
        assert!(stats.eval_duration_ns.is_some());
        assert!(stats.backend_tokens_per_second.is_some());
    }

    #[test]
    fn test_llama_server_timings_parsing() {
        let mut server = Server::new();
        let model_name = "test-model";
        let base_url = server.url();

        // Test specific timing values
        let stream_body = "data: {\"choices\":[{\"delta\":{\"content\":\"test\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":100,\"prompt_ms\":200.5,\"predicted_n\":50,\"predicted_ms\":1000.0}}\n\ndata: [DONE]\n\n";

        let _m = server
            .mock("POST", "/chat/completions")
            .with_status(200)
            .with_body(stream_body)
            .create();

        let model_path = PathBuf::from(model_name);
        let mut model = LlamaServerModel::new_with_base_url(&model_path, &base_url).unwrap();

        let callback = Box::new(|_token: String, _is_first: bool| true);

        let options = GenerationOptions::default();
        let stats = model
            .generate("test prompt", 10, &options, callback)
            .unwrap();

        // Verify timing conversions (ms -> ns)
        assert_eq!(
            stats.prompt_eval_duration_ns,
            Some((200.5 * 1_000_000.0) as u64)
        );
        assert_eq!(stats.eval_duration_ns, Some((1000.0 * 1_000_000.0) as u64));
        assert_eq!(stats.prompt_eval_count, Some(100));
        assert_eq!(stats.eval_count, Some(50));

        // Verify tokens per second calculation: 50 tokens / 1.0 seconds = 50 t/s
        let tps = stats.backend_tokens_per_second.unwrap();
        assert!((tps - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_ollama_generate_with_options() {
        let mut server = Server::new();
        let model_name = "test-model";
        let base_url = server.url();

        let _m = server
            .mock("POST", "/api/generate")
            .with_status(200)
            .with_body_from_request(|req| {
                // Verify the request includes options
                let body: serde_json::Value = serde_json::from_slice(req.body().unwrap()).unwrap();
                assert!(body.get("options").is_some());
                let opts = body.get("options").unwrap();
                assert_eq!(opts.get("temperature").unwrap().as_f64().unwrap(), 0.7);
                assert_eq!(opts.get("top_k").unwrap().as_i64().unwrap(), 50);

                // Return streaming response
                r#"{"model":"test-model","created_at":"2024-01-01T00:00:00Z","response":"hello","done":false}
{"model":"test-model","created_at":"2024-01-01T00:00:00Z","response":" world","done":false}
{"model":"test-model","created_at":"2024-01-01T00:00:00Z","response":"","done":true,"prompt_eval_count":5,"eval_count":2,"eval_duration":100000000}"#
                    .as_bytes()
                    .to_vec()
            })
            .create();

        let model_path = PathBuf::from(model_name);
        let mut model = OllamaModel::new_with_base_url(&model_path, &base_url).unwrap();

        let tokens = Rc::new(RefCell::new(Vec::new()));
        let tokens_clone = tokens.clone();

        let callback = Box::new(move |token: String, _is_first: bool| {
            tokens_clone.borrow_mut().push(token);
            true
        });

        let options = GenerationOptions {
            temperature: Some(0.7),
            top_k: Some(50),
            ..Default::default()
        };

        let stats = model
            .generate("test prompt", 10, &options, callback)
            .unwrap();

        assert_eq!(*tokens.borrow(), vec!["hello", " world"]);
        assert_eq!(stats.prompt_eval_count, Some(5));
        assert_eq!(stats.eval_count, Some(2));
        assert!(stats.backend_tokens_per_second.is_some());
    }
}
