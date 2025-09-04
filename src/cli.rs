use anyhow::Result;
use clap::{Parser, Subcommand, Args, ValueEnum, value_parser};
use std::path::PathBuf;
use log::info;

use crate::benchmark;
use crate::llm_integration::LlmBackend;

/// Hardware Benchmark CLI for Local LLM Inference
#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct CliArgs {
    /// Set the log level (error, warn, info, debug, trace)
    #[clap(short, long, default_value = "info")]
    pub log_level: String,
    
    /// Subcommand to execute
    #[clap(subcommand)]
    pub command: Command,
}

/// CLI subcommands
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run a benchmark
    Benchmark(BenchmarkArgs),
    
    /// Get information about a model
    Info(InfoArgs),
}

/// Arguments for the benchmark command
#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    /// Path to the model file
    #[clap(short, long)]
    pub model_path: PathBuf,
    
    /// LLM backend to use
    #[clap(short = 'b', long, value_enum, default_value_t = LlmBackend::LlamaC)]
    pub mode: LlmBackend,
    
    /// Number of warm-up iterations to run before benchmarking
    #[clap(short, long, default_value_t = 1)]
    pub warmup: u32,
    
    /// Number of benchmark iterations to run
    #[clap(short, long, default_value_t = 3)]
    pub iterations: u32,
    
    /// Prompt to use for generation
    #[clap(short, long, default_value = "Once upon a time")]
    pub prompt: String,
    
    /// Maximum number of tokens to generate
    #[clap(short = 't', long, default_value_t = 50)]
    pub max_tokens: u32,
    
    /// Output format for benchmark results
    #[clap(short, long, value_enum, default_value_t = OutputFormat::Table)]
    pub output: OutputFormat,
    
    /// Collect system metrics during benchmarking
    #[clap(short, long, default_value_t = false)]
    pub system_metrics: bool,
    
    /// Verbosity level (0=minimal, 1=normal, 2=detailed)
    #[clap(short, long, default_value_t = 1)]
    pub verbosity: u8,
    
    /// Controls output randomness (0.0-2.0, default: 1.0)
    #[clap(long = "temperature", value_parser = value_parser!(f32), default_value_t = 1.0)]
    pub temperature: f32,
    
    /// Focus on top K likely tokens (default: 40)
    #[clap(long = "top_k", value_parser = value_parser!(i32), default_value_t = 40)]
    pub top_k: i32,
    
    /// Nucleus sampling probability threshold (default: 0.9)
    #[clap(long = "top_p", value_parser = value_parser!(f32), default_value_t = 0.9)]
    pub top_p: f32,
    
    /// Penalize repeating tokens (default: 1.1)
    #[clap(long = "repeat_penalty", value_parser = value_parser!(f32), default_value_t = 1.1)]
    pub repeat_penalty: f32,
    
    /// Context window size (default: 2048)
    #[clap(long = "context_length", value_parser = value_parser!(i32), default_value_t = 2048)]
    pub context_length: i32,
    
    /// Mirostat mode (0=off, 1=on, default: 0)
    #[clap(long = "mirostat", value_parser = value_parser!(i32), default_value_t = 0)]
    pub mirostat: i32,
}

/// Arguments for the info command
#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Path to the model file
    #[clap(short, long)]
    pub model_path: PathBuf,
    
    /// LLM backend to use
    #[clap(short = 'b', long, value_enum, default_value_t = LlmBackend::LlamaC)]
    pub mode: LlmBackend,
}

/// Output format for benchmark results
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table format
    Table,
    /// Machine-readable JSON format
    Json,
}

/// Parse command line arguments
pub fn parse_args() -> Result<CliArgs> {
    Ok(CliArgs::parse())
}

/// Execute the command specified in the CLI arguments
pub fn execute_command(args: &CliArgs) -> Result<()> {
    match &args.command {
        Command::Benchmark(bench_args) => {
            info!("Running benchmark");
            // We'll handle options in a future implementation
            // For now, just call the benchmark without options
            benchmark::run_benchmark(bench_args)
        },
        Command::Info(info_args) => {
            info!("Getting model information for {}", info_args.model_path.display());
            
            // For now, just display placeholder information
            println!("Model: {}", info_args.model_path.display());
            println!("Backend: {:?}", info_args.mode);
            
            match info_args.mode {
                LlmBackend::LlamaC => {
                    println!("Parameters: ~7B (placeholder)");
                    println!("Context Length: 2048 (placeholder)");
                    println!("Vocabulary Size: 32000 (placeholder)");
                },
                LlmBackend::LlamaRs => {
                    println!("Parameters: ~7B (placeholder)");
                    println!("Context Length: 2048 (placeholder)");
                    println!("Vocabulary Size: 32000 (placeholder)");
                },
                LlmBackend::Ollama => {
                    println!("Using Ollama model: {}", info_args.model_path.display());
                    println!("Note: Actual model parameters depend on the specific model loaded in Ollama");
                }
                LlmBackend::LmStudio => {
                    println!("Using LM Studio model: {}", info_args.model_path.display());
                    println!("Note: Actual model parameters depend on the specific model loaded in LM Studio");
                }
            }
            
            Ok(())
        }
    }
}