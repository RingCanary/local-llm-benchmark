# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Add `CHANGELOG.md` for ongoing release notes.

### Changed
- Migrate crate edition from Rust 2021 to Rust 2024 in `Cargo.toml`.
- Update docs to reflect current backend status (Ollama + LM Studio available, llama-c/llama-rs placeholders).
- Add non-network validation commands (`cargo fmt --all -- --check`, `cargo check`, optional simulated `llama-c` run) to `README.md`.
- Update `AGENTS.md` stack note to Rust 2024.
- Clean duplicate target ignore entry in `.gitignore`.
