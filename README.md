# Hardware Benchmark CLI for Local LLM Inference

## Overview

This CLI application is designed to benchmark hardware performance for running local LLM inference tasks. Built in Rust for memory safety and speed, it will measure performance metrics such as latency, token throughput, and system resource usage while running LLM models. The tool is built to interface with local LLM libraries like [llama.c](https://github.com/ggerganov/llama.cpp), [llama.rs](https://github.com/temporalite/llama.rs), and [Ollama](https://ollama.ai/), using an abstraction layer to support all options.

## Features

- **Command-Line Interface:** Built using a robust CLI crate (Clap) to facilitate argument parsing.
- **Benchmarking Suite:** Executes warm-up and timed benchmarking cycles to measure:
  - Model load time
  - Inference latency (per token and generation overall)
  - Throughput (tokens per second)
- **LLM Integration:** Supports multiple backends:
  - llama.c (via FFI)
  - llama.rs (pure Rust implementation)
  - Ollama (for easy access to a wide range of models)
- **Model Information:** Retrieves and displays detailed model information:
  - Model family/architecture
  - Parameter count
  - Quantization level
  - Context length
- **Hardware Information:** Collects and displays detailed system specifications:
  - CPU model and core count
  - Total system memory
  - Operating system details
  - Disk information
- **System Metrics:** Integrates system metrics collection during benchmarking:
  - CPU usage
  - Memory usage (average and peak)
  - GPU usage (when available)
- **Output Formats:** Results can be presented in human-friendly tables or machine-parsable JSON.
- **Cross-Platform Compatibility:** Uses conditional compilation for platform-specific optimizations.
- **Verbosity Levels:** Supports three verbosity levels for output control.
- **LLM Output Display:** Displays the actual model output from the last iteration, allowing for evaluation of both performance metrics and output quality.

## Project Architecture

```
Project Root
├── Cargo.toml
└── src/
    ├── main.rs             // Entry point: sets up CLI and logging
    ├── cli.rs              // CLI argument parsing (using Clap)
    ├── benchmark.rs        // Core benchmarking logic (timing, iterations, stats)
    ├── llm_integration.rs  // Abstraction for llama.c, llama.rs, and Ollama
    ├── metrics.rs          // System metrics collection
    └── tests/              // Unit tests
└── tests/                  // Integration tests
```

### Module Responsibilities

1. **CLI Layer (src/cli.rs):**
   - Parse arguments, display help and usage.
   - Dispatch commands like `benchmark` or `info`.
   - Handle flags for model path, iteration counts, backend selection, output format, etc.

2. **Benchmark Module (src/benchmark.rs):**
   - Execute a warm-up run followed by multiple benchmark iterations.
   - Time operations with `std::time::Instant`.
   - Compute statistical summaries (mean, standard deviation, etc.).
   - Format and display output.

3. **LLM Integration Module (src/llm_integration.rs):**
   - Abstract interactions with the supported LLM frameworks.
   - Use Rust's FFI for llama.c (isolating unsafe code).
   - Integrate directly with llama.rs functions.
   - Connect to Ollama API for using models hosted by Ollama.
   - Provide a unified API for loading models and generating results based on the selected backend.

4. **System Metrics Module (src/metrics.rs):**
   - Collect system-level metrics (CPU, memory, GPU if applicable) via the sysinfo crate.
   - Associate these metrics with benchmarking results for detailed performance analysis.

## Getting Started

1. **Clone the Repository:**

   ```
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Install Rust:**

   Ensure you have Rust and Cargo installed – visit [rustup.rs](https://rustup.rs) if needed.

3. **Install Ollama (Optional):**

   If you want to use the Ollama backend, install Ollama from [ollama.ai](https://ollama.ai/).

4. **Build the Project:**

   Build in release mode for performance-critical benchmarking:

   ```
   cargo build --release
   ```

5. **Run the Application:**

   Get help and command list:

   ```
   cargo run -- --help
   ```

## Usage Examples

### Basic Benchmarking

```bash
# Benchmark a model using Ollama
cargo run -- benchmark --model-path llama3.2:latest --mode ollama --iterations 3 --warmup 1 --prompt "Write a function to calculate the factorial of a number"

# Benchmark with system metrics collection
cargo run -- benchmark --model-path opencoder:1.5b --mode ollama --iterations 3 --warmup 1 --system-metrics --prompt "Write a function to calculate the factorial of a number"

# Output results in JSON format
cargo run -- benchmark --model-path llama3.2:latest --mode ollama --iterations 3 --output json --prompt "Write a function to calculate the factorial of a number"

# Control output verbosity levels
cargo run -- benchmark --model-path llama3.2:latest --mode ollama --iterations 3 --verbosity 0  # Minimal output
cargo run -- benchmark --model-path llama3.2:latest --mode ollama --iterations 3 --verbosity 1  # Normal output (default)
cargo run -- benchmark --model-path llama3.2:latest --mode ollama --iterations 3 --verbosity 2  # Detailed output with statistics

# Benchmark with model options
cargo run -- benchmark --model-path llama3.2:latest --mode ollama --iterations 3 --temperature 0.8 --top_k 50 --top_p 0.9 --repeat_penalty 1.2 --context_length 2048 --mirostat 1
```

### Model Options

The benchmarking tool supports several model options that influence the behavior of the LLM during inference. These parameters are primarily for the Ollama backend and allow fine-tuning the generation process:

| Option | Description | Range | Default |
|--------|-------------|-------|---------|
| `--temperature` | Controls randomness in token selection. Higher values increase diversity, lower values make output more deterministic. | 0.0 to 2.0 | 1.0 |
| `--top_k` | Limits token selection to the top K most likely tokens. Higher values allow more diversity. | ≥ 1 | 40 |
| `--top_p` | Nucleus sampling threshold. The model considers tokens with cumulative probability up to this value. | 0.0 to 1.0 | 0.9 |
| `--repeat_penalty` | Penalizes repeating tokens to reduce redundant output. Higher values enforce more variation. | 1.0 to 2.0 | 1.1 |
| `--context_length` | Sets the maximum context window size in tokens. | 512 to 4096 | 2048 |
| `--mirostat` | Mirostat sampling algorithm mode (0=disabled, 1=enabled, 2=Mirostat 2.0). | 0 to 2 | 0 |

**Example with Model Options:**

```bash
cargo run -- benchmark --model-path llama3.2:latest --mode ollama --iterations 3 \
  --temperature 0.7 --top_k 50 --top_p 0.95 --repeat_penalty 1.3 --context_length 2048 --mirostat 1 \
  --prompt "Explain the concept of recursion in programming"
```

### Logging

This application writes logs to both the console and a log file named `output.log` located in the project root directory. The logs include timestamped information for debugging and performance analysis.

### Verbosity Levels

The tool supports three verbosity levels for output:

1. **Minimal (0)**: Shows only a condensed summary with the most important metrics:
   - Model name
   - Tokens per second
   - Average token time
   - CPU usage (if system metrics enabled)

2. **Normal (1)**: Default level, shows comprehensive information:
   - Hardware information (CPU, cores, memory, OS)
   - Model information (when available)
   - Performance metrics
   - System metrics (when enabled)

3. **Detailed (2)**: Shows all available information plus statistical data:
   - Everything in Normal level
   - Standard deviation for token times
   - Min/max token times
   - Disk information
   - Standard deviation for CPU and memory usage
   - Additional model metadata

### Sample Output

```
Benchmark Results
=================
Model: llama3.2:latest
Mode: Ollama
Iterations: 2

Hardware Information
====================
CPU:                      11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
CPU Cores:                4
Memory:                   14.58 GB
OS:                       Ubuntu 24.10

Performance Metrics
===================
Avg. Model Load Time:     49 ms
Avg. First Token Time:    4478 ms
Avg. Token Time:          86.13 ms
Avg. Total Gen. Time:     4479 ms
Avg. Tokens Per Second:   11.62

System Metrics (Avg)
===================
CPU Usage:                61.38%
Memory Usage:             32.90 MB
Peak Memory Usage:        33.68 MB

Model Output (Last Iteration)
============================
function factorial(n) {
    if (n === 0 || n === 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}
```

## Implementation Details

### Benchmarking Methodology

The benchmark process follows these steps:

1. **Initialization:** Parse command-line arguments and set up the environment.
2. **Warm-up Phase:** Execute one or more warm-up iterations to prime the system (results discarded).
3. **Benchmark Phase:** Execute the specified number of benchmark iterations.
4. **Metrics Collection:** For each iteration:
   - Measure model load time
   - Measure first token generation time
   - Measure per-token generation time
   - Track total generation time and tokens per second
   - Optionally collect system metrics (CPU, memory usage)
5. **Results Aggregation:** Calculate averages across all iterations.
6. **Output:** Display results in the specified format (table or JSON).

### Benchmark Output Format

After running a benchmark command, the CLI displays the following sections:

#### 1. Process Information (from logs)
- Initialization messages and parameter details (model path, mode, iterations, warmup count)
- System information is logged (CPU model, number of cores) as soon as it's collected

#### 2. Hardware Information
- **CPU:** Model name and specifications
- **CPU Cores:** Number of physical cores
- **Memory:** Total system memory in GB
- **OS:** Operating system name and version
- **GPU:** GPU information (if available)

#### 3. Model Information (when available)
- **Name:** Model identifier
- **Family:** Model architecture family
- **Parameters:** Parameter count in billions
- **Quantization:** Quantization level used
- **Context Length:** Maximum context window size

#### 4. Performance Metrics
- **Avg. Model Load Time:** Time to load the model (ms)
- **Avg. First Token Time:** Time to generate the first token (ms)
- **Avg. Token Time:** Average time per token (ms)
- **Avg. Total Gen. Time:** Total generation time (ms)
- **Avg. Tokens Per Second:** Throughput in tokens per second

#### 5. System Metrics (when enabled)
- **CPU Usage:** Average CPU utilization (%)
- **Memory Usage:** Average memory consumption (MB)
- **Peak Memory Usage:** Maximum memory usage during execution (MB)
- **GPU Usage:** GPU utilization (if available)

#### 6. Model Output (Last Iteration)
- Displays the actual model output from the last iteration, allowing for evaluation of both performance metrics and output quality.

This comprehensive reporting makes it easier to diagnose performance bottlenecks, relate model throughput to hardware capabilities, and generate detailed reports for analysis or CI dashboards.

### JSON Output

When using the `--output json` flag, all the above information is provided in a structured JSON format, making it easy to parse and integrate with other tools or dashboards.

## Future Enhancements

The following features are planned for future releases:

- **GPU Metrics:** Enhance GPU metrics collection for systems with CUDA or other GPU acceleration
- **Historical Tracking:** Implement storage and visualization of benchmark results over time
- **Model Comparison:** Add direct comparison between multiple models in a single run
- **Batch Processing:** Support for batch inference benchmarking
- **Export Formats:** Additional export formats beyond JSON (CSV, HTML reports)
- **Web Dashboard:** A simple web interface for visualizing benchmark results

## Contributing

- Follow established coding standards.
- Run tests locally before committing changes.
- Submit pull requests to feature branches for review.
- Engage in peer code reviews through the established workflow system.

## License

[MIT License]

## Contact

For questions, suggestions, or issues, please open an issue in the GitHub repository.