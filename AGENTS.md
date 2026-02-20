# AGENTS

## Purpose
Local Rust CLI for benchmarking local LLM inference performance (latency, throughput, optional system metrics) across backends.

## Stack
- Rust 2024 + Cargo
- clap (CLI), indicatif/colored (console UX)
- serde/serde_json (JSON), fern/log/chrono (logging)
- sysinfo (system metrics), reqwest blocking (local HTTP API calls)

## Repo Map
- `src/main.rs`: parse args, init logger, dispatch commands
- `src/cli.rs`: `benchmark` and `info` subcommands + flags
- `src/benchmark.rs`: warmup/iterations, timing/stat aggregation, output formatting
- `src/llm_integration.rs`: backend abstraction + Ollama/LM Studio + placeholder llama-c/llama-rs
- `src/metrics.rs`: hardware info and runtime CPU/memory sampling
- `README.md`: usage and examples

## Run / Evaluate
- Build: `cargo build --release`
- Help: `cargo run -- --help`
- Benchmark help: `cargo run -- benchmark --help`
- Example:
  `cargo run -- benchmark --model-path llama3.2:latest --mode ollama --warmup 1 --iterations 3 --prompt "Explain recursion"`
- JSON output:
  `cargo run -- benchmark --model-path llama3.2:latest --mode ollama --iterations 3 --output json`
- Tests: `cargo test`

## Contributor Notes
- Default mode is `llama-c`, which is currently simulated output; use `--mode ollama` or `--mode lm-studio` for real local inference.
- Sampling flags (`--temperature`, `--top_k`, `--top_p`, etc.) are parsed but not applied yet.
- Table output with verbosity > 0 prompts for interactive quality rating; use `--output json` or `--verbosity 0` for CI/non-interactive runs.
- Local endpoints: Ollama `http://localhost:11434`, LM Studio `http://localhost:1234/v1`.
- Runtime logs are written to `output.log` in repo root.
