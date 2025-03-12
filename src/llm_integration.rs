use anyhow::{Result, Context};
use std::path::Path;
use log::{info, debug, error};
use clap::ValueEnum;
use reqwest::blocking::Client;
use serde::{Serialize, Deserialize};
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
}

/// Trait for LLM models
pub trait LlmModel {
    fn generate(&mut self, prompt: &str, max_tokens: usize, callback: TokenCallback) -> Result<()>;
    fn get_model_info(&self) -> Result<Option<ModelInfo>> {
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

/// Create an LLM model based on the specified backend
pub fn create_llm_model(model_path: &Path, backend: LlmBackend) -> Result<Box<dyn LlmModel>> {
    match backend {
        LlmBackend::LlamaC => {
            info!("Creating llama.c model from {}", model_path.display());
            Ok(Box::new(LlamaCModel::new(model_path)?))
        },
        LlmBackend::LlamaRs => {
            info!("Creating llama.rs model from {}", model_path.display());
            Ok(Box::new(LlamaRsModel::new(model_path)?))
        },
        LlmBackend::Ollama => {
            info!("Creating Ollama model from {}", model_path.display());
            Ok(Box::new(OllamaModel::new(model_path)?))
        }
    }
}

/// Implementation for llama.c
pub struct LlamaCModel {
    #[allow(dead_code)]
    model_path: String,
    // FFI context would go here
}

impl LlamaCModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        // In a real implementation, this would load the model via FFI
        debug!("Loading llama.c model from {}", model_path.display());
        
        Ok(LlamaCModel {
            model_path: model_path.display().to_string(),
        })
    }
}

impl LlmModel for LlamaCModel {
    fn generate(&mut self, prompt: &str, max_tokens: usize, mut callback: TokenCallback) -> Result<()> {
        debug!("Generating with llama.c model, prompt: {}, max_tokens: {}", prompt, max_tokens);
        
        // In a real implementation, this would call into llama.c
        // For now, we'll just simulate token generation
        for i in 0..max_tokens.min(10) {
            let token = format!("token_{}", i);
            let is_first = i == 0;
            
            if !callback(token, is_first) {
                break;
            }
        }
        
        Ok(())
    }
}

/// Implementation for llama.rs
pub struct LlamaRsModel {
    #[allow(dead_code)]
    model_path: String,
    // llama.rs model would go here
}

impl LlamaRsModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        // In a real implementation, this would load the model via llama.rs
        debug!("Loading llama.rs model from {}", model_path.display());
        
        Ok(LlamaRsModel {
            model_path: model_path.display().to_string(),
        })
    }
}

impl LlmModel for LlamaRsModel {
    fn generate(&mut self, prompt: &str, max_tokens: usize, mut callback: TokenCallback) -> Result<()> {
        debug!("Generating with llama.rs model, prompt: {}, max_tokens: {}", prompt, max_tokens);
        
        // In a real implementation, this would use llama.rs
        // For now, we'll just simulate token generation
        for i in 0..max_tokens.min(10) {
            let token = format!("token_{}", i);
            let is_first = i == 0;
            
            if !callback(token, is_first) {
                break;
            }
        }
        
        Ok(())
    }
}

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
}

impl OllamaModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        // For Ollama, the "path" is actually the model name
        let model_name = model_path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();
            
        debug!("Using Ollama model: {}", model_name);
        
        // Create HTTP client with reasonable timeout
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;
        
        let mut model = OllamaModel {
            model_name,
            client,
            model_info: None,
        };
        
        // Try to fetch model info
        if let Ok(Some(info)) = model.get_model_info() {
            model.model_info = Some(info);
        }
        
        Ok(model)
    }
    
    fn fetch_model_info(&self) -> Result<OllamaModelInfoResponse> {
        let url = format!("http://localhost:11434/api/show");
        
        let response = self.client.post(&url)
            .json(&serde_json::json!({ "name": self.model_name }))
            .send()
            .context("Failed to send request to Ollama API for model info")?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(|_| "Unknown error".to_string());
            error!("Ollama API error: {} - {}", status, error_text);
            return Err(anyhow::anyhow!("Ollama API error: {} - {}", status, error_text));
        }
        
        let model_info: OllamaModelInfoResponse = response.json()
            .context("Failed to parse Ollama model info response")?;
            
        Ok(model_info)
    }
}

impl LlmModel for OllamaModel {
    fn generate(&mut self, prompt: &str, max_tokens: usize, mut callback: TokenCallback) -> Result<()> {
        debug!("Generating with Ollama model: {}, prompt: {}, max_tokens: {}", 
               self.model_name, prompt, max_tokens);
        
        // Prepare request to Ollama API
        let request = OllamaRequest {
            model: self.model_name.clone(),
            prompt: prompt.to_string(),
            stream: true,
            options: Some(OllamaOptions {
                num_predict: Some(max_tokens as i32),
                ..Default::default()
            }),
        };
        
        // Make API request to Ollama
        let url = "http://localhost:11434/api/generate";
        
        let response = self.client.post(url)
            .json(&request)
            .send()
            .context("Failed to send request to Ollama API")?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(|_| "Unknown error".to_string());
            error!("Ollama API error: {} - {}", status, error_text);
            return Err(anyhow::anyhow!("Ollama API error: {} - {}", status, error_text));
        }
        
        // Process streaming response
        let mut is_first = true;
        for line in response.text()?.lines() {
            if line.is_empty() {
                continue;
            }
            
            // Parse JSON response
            let ollama_response: OllamaResponse = serde_json::from_str(line)
                .context("Failed to parse Ollama API response")?;
            
            // Pass token to callback
            if !callback(ollama_response.response, is_first) {
                break;
            }
            
            is_first = false;
            
            // Check if generation is complete
            if ollama_response.done {
                break;
            }
        }
        
        Ok(())
    }
    
    fn get_model_info(&self) -> Result<Option<ModelInfo>> {
        // If we already have model info, return it
        if let Some(ref info) = self.model_info {
            return Ok(Some(info.clone()));
        }
        
        // Otherwise, try to fetch it
        match self.fetch_model_info() {
            Ok(ollama_info) => {
                // Parse parameter count from string like "7B"
                let parameter_count = ollama_info.details.parameter_size
                    .trim_end_matches(|c: char| !c.is_numeric())
                    .parse::<f64>()
                    .ok();
                
                // Create a clone of the details for metadata
                let details_json = serde_json::to_value(ollama_info.details.clone()).unwrap_or_default();
                
                let model_info = ModelInfo {
                    name: ollama_info.model,
                    family: Some(ollama_info.details.family),
                    parameter_count,
                    quantization: ollama_info.details.quantization_level,
                    context_length: Some(ollama_info.details.context_length),
                    metadata: details_json,
                };
                
                Ok(Some(model_info))
            },
            Err(e) => {
                debug!("Failed to fetch model info: {}", e);
                Ok(None)
            }
        }
    }
}