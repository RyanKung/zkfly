//! Command-line entry point for deterministic `MaleCNS` matrix construction.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use zkfly_matrix::{BuildConfig, BuildProgress, build_matrix_with_progress};

/// Parsed command-line options for one build invocation.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Directory containing the official `MaleCNS` v1.0 Feather files.
    #[arg(long, default_value = "data/raw/male-cns-v1.0")]
    input_dir: PathBuf,

    /// New output directory for the deterministic CSR artifact.
    #[arg(long, default_value = "artifacts/male-cns-v1.0")]
    output_dir: PathBuf,
}

/// Parses options, runs the build, and maps failures to a process exit code.
fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let _ignored = error.print();
            return ExitCode::from(2);
        }
    };
    let config = BuildConfig::new(cli.input_dir, cli.output_dir);
    match build_matrix_with_progress(&config, print_progress) {
        Ok(report) => {
            eprintln!(
                "built {} neurons and {} connections in {}",
                report.neurons,
                report.connections,
                report.output_dir.display()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Prints stable progress milestones without flooding the terminal.
fn print_progress(progress: BuildProgress) {
    match progress {
        BuildProgress::NeuronsSelected { raw_rows, selected } => {
            eprintln!("selected {selected} neurons from {raw_rows} annotation rows");
        }
        BuildProgress::NeurotransmittersApplied { raw_rows } => {
            eprintln!("applied neurotransmitters from {raw_rows} rows");
        }
        BuildProgress::ConnectionRowsScanned { pass, raw_rows } => {
            if raw_rows % 10_000_000 < 100_000 {
                eprintln!("connection pass {pass}: scanned {raw_rows} rows");
            }
        }
        BuildProgress::MatrixValidated { connections } => {
            eprintln!("validated {connections} retained connections");
        }
        BuildProgress::WritingArtifacts => eprintln!("writing deterministic artifacts and hashes"),
    }
}
