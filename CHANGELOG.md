# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Add `CHANGELOG.md` for ongoing release notes.
- Add `llama-server` backend mode (`--mode llama-server`) with OpenAI-compatible API support at `http://localhost:8080/v1`.
- Add `GenerationStats` struct to capture backend-native metrics (prompt/eval durations, token counts, backend TPS).

### Changed
- Migrate crate edition from Rust 2021 to Rust 2024 in `Cargo.toml`.
- Update docs to reflect current backend status (Ollama, LM Studio, llama-server available; llama-c/llama-rs placeholders).
- Add non-network validation commands (`cargo fmt --all -- --check`, `cargo check`, optional simulated `llama-c` run) to `README.md`.
- Update `AGENTS.md` stack note to Rust 2024.
- Clean duplicate target ignore entry in `.gitignore`.
- Switch Ollama and LM Studio streaming to true incremental reading via `BufReader::lines()` instead of `response.text()?.lines()`.
- Thread CLI generation options (`--temperature`, `--top_k`, `--top_p`, `--repeat_penalty`, `--context_length`, `--mirostat`) into backend requests for Ollama, LM Studio, and llama-server.
- Populate benchmark `prompt_eval_duration_ms`, `eval_duration_ms`, `prompt_eval_count`, `eval_count` from backend-native metadata when available.
- Use `backend_tokens_per_second` from backends when available, falling back to wall-clock calculation.
- Update `LlmModel::generate` trait signature to accept `GenerationOptions` and return `GenerationStats`.
