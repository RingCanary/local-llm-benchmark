mod cli;
mod benchmark;
mod llm_integration;
mod metrics;

use anyhow::Result;
use log::info;
use fern;
use chrono::Local;
use std::str::FromStr;

fn main() -> Result<()> {
    // Parse command line arguments
    let args = cli::parse_args()?;
    
    // Initialize logger using fern instead of env_logger
    setup_logger(&args.log_level).expect("Failed to initialize logger");

    info!("Starting LLM Hardware Benchmarker");
    
    // Execute the command
    cli::execute_command(&args)?;
    
    info!("Finished successfully");
    Ok(())
}

// New logger setup function using fern; writes logs to stdout and output.log
fn setup_logger(log_level: &str) -> Result<(), Box<dyn std::error::Error>> {
    let level = log::LevelFilter::from_str(log_level).unwrap_or(log::LevelFilter::Info);
    let log_file = fern::log_file("output.log")?;
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} [{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                message
            ))
        })
        .level(level)
        .chain(std::io::stdout())
        .chain(log_file)
        .apply()?;
    Ok(())
}