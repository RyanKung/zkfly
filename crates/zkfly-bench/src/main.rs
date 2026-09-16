//! Command-line entry point for the first performance-estimation build.

use std::path::PathBuf;

use clap::Parser;

use zkfly_bench::{
    BenchError, BenchmarkConfig, estimate_proof_steps, load_artifact, load_artifact_dimensions,
    run_benchmark,
};

/// Command-line arguments for one reproducible benchmark run.
#[derive(Debug, Parser)]
#[command(name = "zkfly-bench", about = "Benchmark one MaleCNS CSR forward pass")]
struct Arguments {
    /// Directory containing `row_offsets`, columns, counts, and sums.
    #[arg(long, default_value = "artifacts/male-cns-v1.0")]
    artifact_dir: PathBuf,
    /// Untimed warmup executions.
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Timed executions.
    #[arg(long, default_value_t = 10)]
    iterations: usize,
    /// Evaluate only the first N output rows.
    #[arg(long)]
    rows_limit: Option<usize>,
    /// CUDA device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Run only the portable CPU oracle.
    #[arg(long)]
    cpu_only: bool,
    /// Keep the CSR topology and vectors on the GPU across timed executions.
    #[arg(long)]
    device_resident: bool,
    /// Use the two-pass edge-product and row-reduction CUDA variant.
    #[arg(long)]
    edge_parallel: bool,
    /// Use paper step-count mode instead of running the forward benchmark.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    estimate_only: Option<bool>,
    /// Number of game or simulation ticks in the paper execution model.
    #[arg(long)]
    estimate_ticks: Option<usize>,
    /// Number of network updates per tick in the paper execution model.
    #[arg(long, default_value_t = 1)]
    estimate_layers: usize,
    /// Number of graph tiles per network update in the paper execution model.
    #[arg(long, default_value_t = 1)]
    estimate_tiles: usize,
    /// Include the one-time topology registration proof in the total.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    include_topology: Option<bool>,
}

/// Loads the artifact, executes the benchmark, and emits one JSON report.
fn main() -> Result<(), BenchError> {
    let arguments = Arguments::parse();
    if arguments.estimate_only == Some(true) {
        let ticks = arguments.estimate_ticks.ok_or(BenchError::Configuration {
            proposition: "estimate_only requires estimate_ticks",
        })?;
        let dimensions = load_artifact_dimensions(&arguments.artifact_dir)?;
        let estimate = estimate_proof_steps(
            dimensions.neuron_count,
            dimensions.edge_count,
            ticks,
            arguments.estimate_layers,
            arguments.estimate_tiles,
            arguments.include_topology == Some(true),
        )?;
        let json =
            serde_json::to_string_pretty(&estimate).map_err(|_| BenchError::Configuration {
                proposition: "proof step estimate is serializable",
            })?;
        println!("{json}");
        return Ok(());
    }
    let artifact = load_artifact(arguments.artifact_dir)?;
    let config = BenchmarkConfig {
        warmup: arguments.warmup,
        iterations: arguments.iterations,
        rows_limit: arguments.rows_limit,
        device_ordinal: arguments.device,
        cpu_only: arguments.cpu_only,
        device_resident: arguments.device_resident,
        edge_parallel: arguments.edge_parallel,
    };
    let report = run_benchmark(&artifact, config)?;
    let json = serde_json::to_string_pretty(&report).map_err(|_| BenchError::Configuration {
        proposition: "benchmark report is serializable",
    })?;
    println!("{json}");
    Ok(())
}
