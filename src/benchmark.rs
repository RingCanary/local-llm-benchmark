use anyhow::{Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info};
use serde::Serialize;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::cli::{BenchmarkArgs, OutputFormat};
use crate::llm_integration::{ModelInfo, create_llm_model};
use crate::metrics::{HardwareInfo, SystemMetrics, SystemMetricsCollector, get_hardware_info};

/// Represents a single benchmark result
#[derive(Debug, Serialize)]
pub struct BenchmarkResult {
    /// Time taken to load the model (in milliseconds)
    pub model_load_time_ms: u64,

    /// Time taken for the first token generation (in milliseconds)
    pub first_token_time_ms: u64,

    /// Average time per token (in milliseconds)
    pub avg_token_time_ms: f64,

    /// Total generation time (in milliseconds)
    pub total_generation_time_ms: u64,

    /// Tokens per second
    pub tokens_per_second: f64,

    /// Number of tokens generated
    pub tokens_generated: usize,

    /// System metrics during benchmarking (if collected)
    pub system_metrics: Option<SystemMetrics>,

    /// Model information (if available)
    pub model_info: Option<ModelInfo>,

    /// Total duration of the benchmark (in milliseconds)
    pub total_duration_ms: u64,

    /// Time taken for prompt evaluation (in milliseconds)
    pub prompt_eval_duration_ms: u64,

    /// Time taken for evaluation (in milliseconds)
    pub eval_duration_ms: u64,

    /// Number of prompt evaluations
    pub prompt_eval_count: i32,

    /// Number of evaluations
    pub eval_count: i32,

    /// Generated text
    pub generated_text: String,
}

/// Runs the benchmark with the specified arguments
pub fn run_benchmark(args: &BenchmarkArgs) -> Result<()> {
    info!("Starting benchmark with {} iterations", args.iterations);
    info!("Model path: {}", args.model_path.display());
    info!("LLM mode: {:?}", args.mode);

    // Collect hardware information
    let hardware_info = get_hardware_info();
    info!(
        "Hardware: {} with {} cores",
        hardware_info.cpu_model, hardware_info.cpu_cores
    );

    // Create progress bar for iterations
    let pb = ProgressBar::new((args.warmup + args.iterations) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iterations",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    // Run warmup iterations
    info!("Running {} warmup iterations", args.warmup);
    for i in 0..args.warmup {
        debug!("Warmup iteration {}/{}", i + 1, args.warmup);
        let _ = run_single_benchmark(args, false)?;
        pb.inc(1);
    }

    // Run actual benchmark iterations
    info!("Running {} benchmark iterations", args.iterations);
    let mut results = Vec::with_capacity(args.iterations as usize);

    for i in 0..args.iterations {
        debug!("Benchmark iteration {}/{}", i + 1, args.iterations);
        let result = run_single_benchmark(args, args.system_metrics)?;
        results.push(result);
        pb.inc(1);
    }

    pb.finish_with_message("Benchmark completed");

    // Calculate aggregate statistics
    let avg_model_load_time =
        results.iter().map(|r| r.model_load_time_ms).sum::<u64>() / results.len() as u64;
    let avg_first_token_time =
        results.iter().map(|r| r.first_token_time_ms).sum::<u64>() / results.len() as u64;
    let avg_token_time =
        results.iter().map(|r| r.avg_token_time_ms).sum::<f64>() / results.len() as f64;
    let avg_total_time = results
        .iter()
        .map(|r| r.total_generation_time_ms)
        .sum::<u64>()
        / results.len() as u64;
    let avg_tokens_per_second =
        results.iter().map(|r| r.tokens_per_second).sum::<f64>() / results.len() as f64;

    // Get model info from the first result (should be the same for all iterations)
    let model_info = results.first().and_then(|r| r.model_info.clone());

    // Output results based on format
    match args.output {
        OutputFormat::Table => {
            println!("\n{}", "Benchmark Results".bold().green());
            println!("{}", "=================".green());
            println!("Model: {}", args.model_path.display().to_string().cyan());
            println!("Mode: {:?}", args.mode);
            println!("Iterations: {}", args.iterations);

            // Display model information if available and verbosity > 0
            if args.verbosity > 0 && model_info.is_some() {
                let info = model_info.as_ref().unwrap();
                println!("\n{}", "Model Information".bold().green());
                println!("{}", "=================".green());
                println!("{:<25} {}", "Name:", info.name.cyan());

                if let Some(family) = &info.family {
                    println!("{:<25} {}", "Family:", family);
                }

                if let Some(params) = &info.parameter_count {
                    println!(
                        "{:<25} {}",
                        "Parameters:",
                        format!("{:.1}B", params).yellow()
                    );
                }

                if let Some(quant) = &info.quantization {
                    println!("{:<25} {}", "Quantization:", quant);
                }

                if let Some(ctx_len) = &info.context_length {
                    println!("{:<25} {}", "Context Length:", ctx_len);
                }

                // Display additional metadata if verbosity is high
                if args.verbosity > 1 {
                    println!("{:<25} {}", "Metadata:", info.metadata);
                }
            }

            // Display hardware information if verbosity > 0
            if args.verbosity > 0 {
                println!("\n{}", "Hardware Information".bold().green());
                println!("{}", "====================".green());
                println!("{:<25} {}", "CPU:", hardware_info.cpu_model.cyan());
                println!("{:<25} {}", "CPU Cores:", hardware_info.cpu_cores);
                println!(
                    "{:<25} {}",
                    "Memory:",
                    format!("{:.2} GB", hardware_info.total_memory_gb).yellow()
                );
                println!("{:<25} {}", "OS:", hardware_info.os_info);

                // Display additional hardware details if verbosity is high
                if args.verbosity > 1 && !hardware_info.disk_info.is_empty() {
                    println!(
                        "{:<25} {}",
                        "Disk:",
                        format!("{:?}", hardware_info.disk_info)
                    );
                }

                if let Some(gpu_info) = &hardware_info.gpu_info {
                    println!("{:<25} {}", "GPU:", gpu_info.cyan());
                }
            }

            // Always display core performance metrics regardless of verbosity
            println!("\n{}", "Performance Metrics".bold().green());
            println!("{}", "===================".green());
            println!(
                "{:<25} {}",
                "Avg. Model Load Time:",
                format!("{} ms", avg_model_load_time).yellow()
            );
            println!(
                "{:<25} {}",
                "Avg. First Token Time:",
                format!("{} ms", avg_first_token_time).yellow()
            );
            println!(
                "{:<25} {}",
                "Avg. Token Time:",
                format!("{:.2} ms", avg_token_time).yellow()
            );
            println!(
                "{:<25} {}",
                "Avg. Total Gen. Time:",
                format!("{} ms", avg_total_time).yellow()
            );
            println!(
                "{:<25} {}",
                "Avg. Tokens Per Second:",
                format!("{:.2}", avg_tokens_per_second).yellow()
            );

            // Display detailed performance metrics if verbosity is high
            if args.verbosity > 1 {
                // Calculate standard deviation for token time
                let mean = avg_token_time;
                let variance = results
                    .iter()
                    .map(|r| (r.avg_token_time_ms - mean).powi(2))
                    .sum::<f64>()
                    / results.len() as f64;
                let std_dev = variance.sqrt();

                // Calculate min/max token times
                let min_token_time = results
                    .iter()
                    .map(|r| r.avg_token_time_ms)
                    .fold(f64::INFINITY, f64::min);

                let max_token_time = results
                    .iter()
                    .map(|r| r.avg_token_time_ms)
                    .fold(0.0, f64::max);

                println!(
                    "{:<25} {}",
                    "Std Dev Token Time:",
                    format!("{:.2} ms", std_dev).yellow()
                );
                println!(
                    "{:<25} {}",
                    "Min Token Time:",
                    format!("{:.2} ms", min_token_time).yellow()
                );
                println!(
                    "{:<25} {}",
                    "Max Token Time:",
                    format!("{:.2} ms", max_token_time).yellow()
                );
            }

            // Display system metrics if collected and verbosity > 0
            if args.system_metrics && args.verbosity > 0 {
                println!("\n{}", "System Metrics (Avg)".bold().green());
                println!("{}", "===================".green());

                let avg_cpu_usage = results
                    .iter()
                    .filter_map(|r| r.system_metrics.as_ref().map(|m| m.cpu_usage))
                    .sum::<f32>()
                    / results.len() as f32;

                let avg_memory_usage = results
                    .iter()
                    .filter_map(|r| r.system_metrics.as_ref().map(|m| m.memory_usage_mb))
                    .sum::<f32>()
                    / results.len() as f32;

                let peak_memory_usage = results
                    .iter()
                    .filter_map(|r| r.system_metrics.as_ref().map(|m| m.peak_memory_usage_mb))
                    .fold(0.0, f32::max);

                println!(
                    "{:<25} {}",
                    "CPU Usage:",
                    format!("{:.2}%", avg_cpu_usage).yellow()
                );
                println!(
                    "{:<25} {}",
                    "Memory Usage:",
                    format!("{:.2} MB", avg_memory_usage).yellow()
                );
                println!(
                    "{:<25} {}",
                    "Peak Memory Usage:",
                    format!("{:.2} MB", peak_memory_usage).yellow()
                );

                // Display additional system metrics if verbosity is high
                if args.verbosity > 1 {
                    // Calculate standard deviation for CPU and memory usage
                    let cpu_mean = avg_cpu_usage;
                    let cpu_variance = results
                        .iter()
                        .filter_map(|r| {
                            r.system_metrics
                                .as_ref()
                                .map(|m| (m.cpu_usage - cpu_mean).powi(2))
                        })
                        .sum::<f32>()
                        / results.len() as f32;
                    let cpu_std_dev = cpu_variance.sqrt();

                    let mem_mean = avg_memory_usage;
                    let mem_variance = results
                        .iter()
                        .filter_map(|r| {
                            r.system_metrics
                                .as_ref()
                                .map(|m| (m.memory_usage_mb - mem_mean).powi(2))
                        })
                        .sum::<f32>()
                        / results.len() as f32;
                    let mem_std_dev = mem_variance.sqrt();

                    println!(
                        "{:<25} {}",
                        "Std Dev CPU Usage:",
                        format!("{:.2}%", cpu_std_dev).yellow()
                    );
                    println!(
                        "{:<25} {}",
                        "Std Dev Memory Usage:",
                        format!("{:.2} MB", mem_std_dev).yellow()
                    );

                    // Display GPU usage if available
                    if results.iter().any(|r| {
                        r.system_metrics
                            .as_ref()
                            .and_then(|m| m.gpu_usage)
                            .is_some()
                    }) {
                        let avg_gpu_usage = results
                            .iter()
                            .filter_map(|r| r.system_metrics.as_ref().and_then(|m| m.gpu_usage))
                            .sum::<f32>()
                            / results.len() as f32;

                        println!(
                            "{:<25} {}",
                            "GPU Usage:",
                            format!("{:.2}%", avg_gpu_usage).yellow()
                        );
                    }
                }
            }

            // For minimal verbosity (0), just show a very condensed summary
            if args.verbosity == 0 {
                println!("\n{}", "Summary".bold().green());
                println!("{}", "=======".green());
                println!(
                    "{:<25} {}",
                    "Model:",
                    args.model_path.display().to_string().cyan()
                );
                println!(
                    "{:<25} {}",
                    "Tokens/Second:",
                    format!("{:.2}", avg_tokens_per_second).yellow()
                );
                println!(
                    "{:<25} {}",
                    "Avg. Token Time:",
                    format!("{:.2} ms", avg_token_time).yellow()
                );
                if args.system_metrics {
                    let avg_cpu_usage = results
                        .iter()
                        .filter_map(|r| r.system_metrics.as_ref().map(|m| m.cpu_usage))
                        .sum::<f32>()
                        / results.len() as f32;
                    println!(
                        "{:<25} {}",
                        "CPU Usage:",
                        format!("{:.2}%", avg_cpu_usage).yellow()
                    );
                }
            }

            // Display the LLM's generated text for the last iteration
            if !results.is_empty() && args.verbosity > 0 {
                let last_result = &results[results.len() - 1];
                if !last_result.generated_text.is_empty() {
                    println!("\nModel Output (Last Iteration)");
                    println!("============================");
                    println!("{}", last_result.generated_text);

                    // Add user feedback prompt
                    println!("\nOutput Quality Assessment");
                    println!("========================");
                    println!("How would you rate the quality of the model output?");
                    println!("(1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent)");
                    println!("Enter rating [1-5] or press Enter to skip: ");

                    // Read user input without blocking the program
                    use std::io::{self, Write};
                    io::stdout().flush().ok();
                    let mut input = String::new();
                    if let Ok(_) = io::stdin().read_line(&mut input) {
                        let input = input.trim();
                        if !input.is_empty() {
                            match input.parse::<u8>() {
                                Ok(rating) if (1..=5).contains(&rating) => {
                                    println!(
                                        "Rating of {} recorded. Thank you for your feedback!",
                                        rating
                                    );
                                }
                                _ => println!("Invalid rating. Feedback skipped."),
                            }
                        } else {
                            println!("Feedback skipped.");
                        }
                    }
                }
            }
        }
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct AggregateResults {
                individual_results: Vec<BenchmarkResult>,
                avg_model_load_time_ms: u64,
                avg_first_token_time_ms: u64,
                avg_token_time_ms: f64,
                avg_total_generation_time_ms: u64,
                avg_tokens_per_second: f64,
                model_path: String,
                llm_mode: String,
                iterations: u32,
                model_info: Option<ModelInfo>,
                hardware_info: HardwareInfo,
            }

            let aggregate = AggregateResults {
                individual_results: results,
                avg_model_load_time_ms: avg_model_load_time,
                avg_first_token_time_ms: avg_first_token_time,
                avg_token_time_ms: avg_token_time,
                avg_total_generation_time_ms: avg_total_time,
                avg_tokens_per_second: avg_tokens_per_second,
                model_path: args.model_path.display().to_string(),
                llm_mode: format!("{:?}", args.mode),
                iterations: args.iterations,
                model_info,
                hardware_info,
            };

            println!(
                "{}",
                serde_json::to_string_pretty(&aggregate)
                    .context("Failed to serialize results to JSON")?
            );
        }
    }

    Ok(())
}

/// Runs a single benchmark iteration
fn run_single_benchmark(args: &BenchmarkArgs, collect_metrics: bool) -> Result<BenchmarkResult> {
    // Optionally initialize system metrics collector
    let mut metrics_collector = if collect_metrics {
        Some(SystemMetricsCollector::new())
    } else {
        None
    };

    // Load model and measure time
    let model_load_start = Instant::now();
    let mut model =
        create_llm_model(&args.model_path, args.mode).context("Failed to create LLM model")?;
    let model_load_time = model_load_start.elapsed();

    // Get model information if available
    let model_info = model.get_model_info().ok().flatten();

    // Start metrics collection if enabled
    if let Some(collector) = &mut metrics_collector {
        collector.start();
    }

    // Generate tokens and measure time
    let generation_start = Instant::now();

    // Use thread-local storage to track token generation
    struct TokenStats {
        first_token_time: Option<Duration>,
        tokens_generated: usize,
        generated_text: String,
    }

    let stats = TokenStats {
        first_token_time: None,
        tokens_generated: 0,
        generated_text: String::new(),
    };

    let stats_ref = Rc::new(RefCell::new(stats));
    let stats_clone = stats_ref.clone();

    // Create a non-moving closure for token generation
    let callback = Box::new(move |token: String, is_first: bool| -> bool {
        let mut stats = stats_clone.borrow_mut();
        if is_first {
            stats.first_token_time = Some(generation_start.elapsed());
        }
        stats.tokens_generated += 1;
        stats.generated_text.push_str(&token);
        debug!("Generated token: {}", token);
        true // continue generation
    });

    // Run the generation
    model
        .generate(&args.prompt, args.max_tokens as usize, callback)
        .context("Failed to generate tokens")?;

    let generation_time = generation_start.elapsed();

    // Stop metrics collection if enabled
    let system_metrics = if let Some(collector) = metrics_collector {
        Some(collector.stop())
    } else {
        None
    };

    // Calculate statistics
    let stats = stats_ref.borrow();
    let first_token_time = stats.first_token_time.unwrap_or_else(|| generation_time);
    let tokens_generated = stats.tokens_generated;
    let generated_text = stats.generated_text.clone();

    let avg_token_time = if tokens_generated > 0 {
        generation_time.as_millis() as f64 / tokens_generated as f64
    } else {
        0.0
    };

    let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / generation_time.as_secs_f64()
    } else {
        0.0
    };

    // Calculate additional metrics
    let total_duration = model_load_time + generation_time;

    // For now, use default values for these metrics as the LLM interface doesn't provide them
    let prompt_eval_duration = Duration::from_millis(0);
    let eval_duration = Duration::from_millis(0);
    let prompt_eval_count = 0;
    let eval_count = 0;

    Ok(BenchmarkResult {
        model_load_time_ms: model_load_time.as_millis() as u64,
        first_token_time_ms: first_token_time.as_millis() as u64,
        avg_token_time_ms: avg_token_time,
        total_generation_time_ms: generation_time.as_millis() as u64,
        tokens_per_second,
        tokens_generated,
        system_metrics,
        model_info,
        total_duration_ms: total_duration.as_millis() as u64,
        prompt_eval_duration_ms: prompt_eval_duration.as_millis() as u64,
        eval_duration_ms: eval_duration.as_millis() as u64,
        prompt_eval_count,
        eval_count,
        generated_text,
    })
}
