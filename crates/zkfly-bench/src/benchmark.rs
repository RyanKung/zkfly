//! Benchmark orchestration and the portable CPU oracle.

use std::time::{Duration, Instant};

use serde::Serialize;

use crate::artifact::{ArtifactSummary, CsrArtifact};
use crate::error::BenchError;

/// Parameters that define one reproducible benchmark run.
#[derive(Clone, Copy, Debug)]
pub struct BenchmarkConfig {
    /// Number of untimed GPU warmup executions.
    pub warmup: usize,
    /// Number of timed executions.
    pub iterations: usize,
    /// Optional prefix of rows to evaluate, useful for smoke tests.
    pub rows_limit: Option<usize>,
    /// CUDA device ordinal selected by the caller.
    pub device_ordinal: usize,
    /// Do not attempt CUDA, even when the binary contains the backend.
    pub cpu_only: bool,
    /// Keep the topology, weights, input, and output allocations on the GPU.
    pub device_resident: bool,
    /// Use the two-pass edge-product and row-reduction CUDA variant.
    pub edge_parallel: bool,
}

impl Default for BenchmarkConfig {
    /// Returns a conservative configuration suitable for a first remote run.
    fn default() -> Self {
        Self {
            warmup: 3,
            iterations: 10,
            rows_limit: None,
            device_ordinal: 0,
            cpu_only: false,
            device_resident: false,
            edge_parallel: false,
        }
    }
}

/// Measured timings and correctness data for one benchmark invocation.
#[derive(Debug, Serialize)]
pub struct BenchmarkReport {
    /// Matrix dimensions and byte footprint.
    pub artifact: ArtifactSummary,
    /// Number of rows evaluated in each execution.
    pub rows_benchmarked: usize,
    /// Deterministic checksum of the CPU oracle output.
    pub cpu_checksum: f64,
    /// CPU wall-clock duration for one oracle execution in milliseconds.
    pub cpu_ms: f64,
    /// Whether this report used the CUDA backend.
    pub cuda: bool,
    /// Whether CUDA buffers remained allocated across timed executions.
    pub device_resident: bool,
    /// Whether the two-pass edge-parallel CUDA variant was selected.
    pub edge_parallel: bool,
    /// Average host-to-device transfer time in milliseconds, when CUDA ran.
    pub upload_ms: Option<f64>,
    /// Average kernel plus device-to-host time in milliseconds, when CUDA ran.
    pub compute_download_ms: Option<f64>,
    /// Average GPU kernel time from CUDA events, when resident mode ran.
    pub kernel_ms: Option<f64>,
    /// Average device-to-host time in milliseconds, when resident mode ran.
    pub download_ms: Option<f64>,
    /// Average end-to-end CUDA execution time in milliseconds, when CUDA ran.
    pub end_to_end_ms: Option<f64>,
    /// Absolute error against the CPU oracle, when CUDA ran.
    pub max_abs_error: Option<f32>,
    /// Effective edge visits per second, based on end-to-end time.
    pub edges_per_second: Option<f64>,
    /// Approximate matrix and vector bytes moved per second.
    pub estimated_gib_per_second: Option<f64>,
}

/// Runs the CPU oracle and, unless disabled, the CUDA implementation.
///
/// The input vector is generated deterministically from neuron indices. Every
/// timed execution reuses the same vector and topology, so differences between
/// machines are attributable to implementation and hardware rather than input
/// sampling. The first benchmark intentionally uses identity activation.
///
/// # Errors
///
/// Returns [`BenchError::Configuration`] for zero iterations or an invalid row
/// limit, and forwards CUDA errors without falling back to CPU silently.
pub fn run_benchmark(
    artifact: &CsrArtifact,
    config: BenchmarkConfig,
) -> Result<BenchmarkReport, BenchError> {
    validate_config(artifact, config)?;
    let rows = config.rows_limit.unwrap_or(artifact.neuron_count);
    let input = deterministic_input(artifact.neuron_count);
    let cpu_start = Instant::now();
    let cpu_output = csr_spmv_cpu(artifact, &input, rows)?;
    let cpu_ms = duration_ms(cpu_start.elapsed());
    let cpu_checksum = checksum(&cpu_output);
    let summary = artifact.summary();
    if config.cpu_only {
        return Ok(cpu_report(
            summary,
            rows,
            cpu_checksum,
            cpu_ms,
            config.edge_parallel,
        ));
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    {
        return cuda_report(
            artifact,
            &input,
            &cpu_output,
            rows,
            config,
            summary,
            cpu_checksum,
            cpu_ms,
        );
    }
    #[cfg(not(all(feature = "cuda", target_os = "linux")))]
    {
        let _ = (artifact, input, cpu_output, rows, config);
        Err(BenchError::CudaUnavailable)
    }
}

/// Validates benchmark mode relationships before doing CPU or GPU work.
fn validate_config(artifact: &CsrArtifact, config: BenchmarkConfig) -> Result<(), BenchError> {
    if config.iterations == 0 {
        return Err(BenchError::Configuration {
            proposition: "iterations must be nonzero",
        });
    }
    let rows = config.rows_limit.unwrap_or(artifact.neuron_count);
    if rows == 0 || rows > artifact.neuron_count {
        return Err(BenchError::Configuration {
            proposition: "rows_limit must be between one and the neuron count",
        });
    }
    if config.edge_parallel && !config.device_resident {
        return Err(BenchError::Configuration {
            proposition: "edge_parallel requires device_resident",
        });
    }
    if config.edge_parallel && artifact.column_indices.is_empty() {
        return Err(BenchError::Configuration {
            proposition: "edge_parallel requires at least one edge",
        });
    }
    if config.edge_parallel && config.rows_limit.is_some() {
        return Err(BenchError::Configuration {
            proposition: "edge_parallel requires the complete row set",
        });
    }
    Ok(())
}

/// Builds the report for a CPU-only invocation.
fn cpu_report(
    artifact: ArtifactSummary,
    rows: usize,
    cpu_checksum: f64,
    cpu_ms: f64,
    edge_parallel: bool,
) -> BenchmarkReport {
    BenchmarkReport {
        artifact,
        rows_benchmarked: rows,
        cpu_checksum,
        cpu_ms,
        cuda: false,
        device_resident: false,
        edge_parallel,
        upload_ms: None,
        compute_download_ms: None,
        kernel_ms: None,
        download_ms: None,
        end_to_end_ms: None,
        max_abs_error: None,
        edges_per_second: None,
        estimated_gib_per_second: None,
    }
}

/// Converts one CUDA measurement into the stable JSON report.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_report(
    artifact: &CsrArtifact,
    input: &[f32],
    cpu_output: &[f32],
    rows: usize,
    config: BenchmarkConfig,
    summary: ArtifactSummary,
    cpu_checksum: f64,
    cpu_ms: f64,
) -> Result<BenchmarkReport, BenchError> {
    let iteration_count =
        u32::try_from(config.iterations).map_err(|_| BenchError::Configuration {
            proposition: "iterations fit benchmark report arithmetic",
        })?;
    let measurement = crate::cuda_runtime::run_cuda(artifact, input, cpu_output, rows, config)?;
    let end_to_end_seconds = measurement.end_to_end.as_secs_f64() / f64::from(iteration_count);
    let edge_visits = edge_visits(artifact, rows)?;
    let bytes = transferred_bytes(artifact, rows, config.edge_parallel)?;
    let edge_visits_u64 = u64::try_from(edge_visits).map_err(|_| BenchError::Configuration {
        proposition: "edge count fits benchmark report arithmetic",
    })?;
    let upload_ms = if measurement.device_resident {
        duration_ms(measurement.upload)
    } else {
        duration_ms(measurement.upload) / f64::from(iteration_count)
    };
    let kernel_ms = measurement
        .kernel
        .map(|duration| duration_ms(duration) / f64::from(iteration_count));
    let download_ms = measurement
        .download
        .map(|duration| duration_ms(duration) / f64::from(iteration_count));
    Ok(BenchmarkReport {
        artifact: summary,
        rows_benchmarked: rows,
        cpu_checksum,
        cpu_ms,
        cuda: true,
        device_resident: measurement.device_resident,
        edge_parallel: config.edge_parallel,
        upload_ms: Some(upload_ms),
        compute_download_ms: Some(
            duration_ms(measurement.compute_download) / f64::from(iteration_count),
        ),
        kernel_ms,
        download_ms,
        end_to_end_ms: Some(duration_ms(measurement.end_to_end) / f64::from(iteration_count)),
        max_abs_error: Some(measurement.max_abs_error),
        edges_per_second: Some(report_f64(edge_visits_u64) / end_to_end_seconds),
        estimated_gib_per_second: Some(report_f64(bytes) / end_to_end_seconds / 2_f64.powi(30)),
    })
}

/// Computes one row-major sparse matrix/vector product with f32 arithmetic.
pub(crate) fn csr_spmv_cpu(
    artifact: &CsrArtifact,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>, BenchError> {
    let mut output = vec![0.0; rows];
    for row in 0..rows {
        let start_value = *artifact
            .row_offsets
            .get(row)
            .ok_or(BenchError::Configuration {
                proposition: "row limit has a CSR boundary",
            })?;
        let start = usize::try_from(start_value).map_err(|_| BenchError::Configuration {
            proposition: "CSR offset fits host index type",
        })?;
        let end_value = *artifact
            .row_offsets
            .get(row + 1)
            .ok_or(BenchError::Configuration {
                proposition: "row limit has a terminal CSR boundary",
            })?;
        let end = usize::try_from(end_value).map_err(|_| BenchError::Configuration {
            proposition: "CSR offset fits host index type",
        })?;
        let mut acc = 0.0_f32;
        for edge in start..end {
            let column_value =
                *artifact
                    .column_indices
                    .get(edge)
                    .ok_or(BenchError::Configuration {
                        proposition: "CSR column exists for every row edge",
                    })?;
            let column = usize::try_from(column_value).map_err(|_| BenchError::Configuration {
                proposition: "CSR column fits host index type",
            })?;
            let weight = *artifact
                .weights
                .get(edge)
                .ok_or(BenchError::Configuration {
                    proposition: "CSR weight exists for every row edge",
                })?;
            let value = *input.get(column).ok_or(BenchError::Configuration {
                proposition: "input contains every referenced column",
            })?;
            acc += weight * value;
        }
        if let Some(destination) = output.get_mut(row) {
            *destination = acc;
        }
    }
    Ok(output)
}

/// Generates a stable, nonzero input without a random-number dependency.
#[allow(clippy::manual_unwrap_or, clippy::manual_unwrap_or_default)]
fn deterministic_input(neurons: usize) -> Vec<f32> {
    (0..neurons)
        .map(|index| {
            let bucket = match u16::try_from(index % 2048) {
                Ok(value) => value,
                Err(_) => 0,
            };
            f32::from(bucket) / 1024.0 - 1.0
        })
        .collect()
}

/// Counts edges visited by the selected output-row prefix.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn edge_visits(artifact: &CsrArtifact, rows: usize) -> Result<usize, BenchError> {
    let start_value = *artifact
        .row_offsets
        .first()
        .ok_or(BenchError::Configuration {
            proposition: "CSR has a first boundary",
        })?;
    let start = usize::try_from(start_value).map_err(|_| BenchError::Configuration {
        proposition: "CSR offset fits host index type",
    })?;
    let end_value = *artifact
        .row_offsets
        .get(rows)
        .ok_or(BenchError::Configuration {
            proposition: "CSR has the selected terminal boundary",
        })?;
    let end = usize::try_from(end_value).map_err(|_| BenchError::Configuration {
        proposition: "CSR offset fits host index type",
    })?;
    end.checked_sub(start).ok_or(BenchError::Configuration {
        proposition: "CSR boundaries are ordered",
    })
}

/// Estimates bytes touched by one selected-row execution.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn transferred_bytes(
    artifact: &CsrArtifact,
    rows: usize,
    edge_parallel: bool,
) -> Result<u64, BenchError> {
    let edges = edge_visits(artifact, rows)?;
    let edge_bytes = edges
        .checked_mul(std::mem::size_of::<u32>() * 2 + std::mem::size_of::<f32>())
        .ok_or(BenchError::Configuration {
            proposition: "byte estimate fits usize",
        })?;
    let partial_bytes = if edge_parallel {
        edges
            .checked_mul(std::mem::size_of::<f32>() * 2)
            .ok_or(BenchError::Configuration {
                proposition: "partial byte estimate fits usize",
            })?
    } else {
        0
    };
    let vector_bytes = artifact
        .neuron_count
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| bytes.checked_add(rows.saturating_mul(std::mem::size_of::<f32>())))
        .ok_or(BenchError::Configuration {
            proposition: "vector byte estimate fits usize",
        })?;
    u64::try_from(
        edge_bytes
            .saturating_add(partial_bytes)
            .saturating_add(vector_bytes),
    )
    .map_err(|_| BenchError::Configuration {
        proposition: "byte estimate fits u64",
    })
}

/// Computes a stable f64 checksum without changing the f32 execution oracle.
fn checksum(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value)).sum()
}

/// Converts a duration to milliseconds as an f64 for JSON reporting.
fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// Converts a byte or edge count to the floating-point report unit.
#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(clippy::cast_precision_loss)]
fn report_f64(value: u64) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::{BenchmarkConfig, csr_spmv_cpu, run_benchmark};
    use crate::artifact::CsrArtifact;
    use crate::error::BenchError;
    use zkfly_commitment::Commitment;

    #[test]
    fn cpu_oracle_preserves_csr_row_order() {
        let artifact = CsrArtifact {
            neuron_count: 3,
            row_offsets: vec![0, 2, 2, 3],
            column_indices: vec![2, 0, 1],
            weights: vec![0.5, -1.0, 2.0],
            topology_commitment: Commitment::from_bytes([0; 32]),
        };
        let input = vec![1.0, 2.0, 4.0];
        let result = csr_spmv_cpu(&artifact, &input, 3);
        assert!(result.is_ok());
        if let Ok(output) = result {
            assert_eq!(output, vec![1.0, 0.0, 4.0]);
        }
    }

    #[test]
    fn edge_parallel_requires_resident_buffers() {
        let artifact = CsrArtifact {
            neuron_count: 1,
            row_offsets: vec![0, 1],
            column_indices: vec![0],
            weights: vec![1.0],
            topology_commitment: Commitment::from_bytes([0; 32]),
        };
        let result = run_benchmark(
            &artifact,
            BenchmarkConfig {
                warmup: 0,
                iterations: 1,
                rows_limit: None,
                device_ordinal: 0,
                cpu_only: true,
                device_resident: false,
                edge_parallel: true,
            },
        );
        assert!(matches!(
            result,
            Err(BenchError::Configuration {
                proposition: "edge_parallel requires device_resident"
            })
        ));
    }
}
