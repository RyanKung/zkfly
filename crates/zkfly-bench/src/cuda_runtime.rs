//! Safe host adapter for the cuda-oxide generated CSR kernel.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{fs, path::PathBuf};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, thread};

use crate::artifact::CsrArtifact;
use crate::benchmark::BenchmarkConfig;
use crate::error::BenchError;

/// Number of threads in one CUDA block for the row-per-thread baseline.
const BLOCK_SIZE: u32 = 256;

/// Timings and correctness result collected from one CUDA benchmark.
pub(crate) struct CudaMeasurement {
    /// Aggregate host-to-device transfer time.
    pub(crate) upload: Duration,
    /// Aggregate kernel and device-to-host time.
    pub(crate) compute_download: Duration,
    /// Aggregate end-to-end time, including both phases.
    pub(crate) end_to_end: Duration,
    /// Aggregate event-measured kernel time in resident mode.
    pub(crate) kernel: Option<Duration>,
    /// Aggregate device-to-host time in resident mode.
    pub(crate) download: Option<Duration>,
    /// Maximum absolute difference from the CPU f32 oracle.
    pub(crate) max_abs_error: f32,
    /// Whether device allocations survived all timed executions.
    pub(crate) device_resident: bool,
}

/// A persistent CUDA context and generated kernel module.
pub(crate) struct CudaSpmv {
    /// Device context retained across warmups and timed iterations.
    context: Arc<CudaContext>,
    /// Typed module generated from the Rust kernel below.
    module: kernels::LoadedModule,
    /// Serializes context binding and buffer teardown within this MVP.
    execution_lock: Mutex<()>,
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Computes one CSR output row per CUDA thread.
    #[kernel]
    #[launch_bounds(256)]
    pub fn csr_spmv(
        row_offsets: &[u32],
        column_indices: &[u32],
        weights: &[f32],
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let row = thread::index_1d();
        let row_index = row.get();
        let Some(destination) = output.get_mut(row) else {
            return;
        };
        let Some(start) = row_offsets.get(row_index).copied() else {
            return;
        };
        let Some(next_row) = row_index.checked_add(1) else {
            return;
        };
        let Some(end) = row_offsets.get(next_row).copied() else {
            return;
        };
        let mut edge = start;
        let mut accumulator = 0.0_f32;
        while edge < end {
            let Some(edge_index) = usize::try_from(edge).ok() else {
                return;
            };
            let Some(column) = column_indices.get(edge_index).copied() else {
                return;
            };
            let Some(weight) = weights.get(edge_index).copied() else {
                return;
            };
            let Some(column_index) = usize::try_from(column).ok() else {
                return;
            };
            let Some(value) = input.get(column_index).copied() else {
                return;
            };
            accumulator += weight * value;
            let Some(next_edge) = edge.checked_add(1) else {
                return;
            };
            edge = next_edge;
        }
        *destination = accumulator;
    }

    /// Computes one weighted edge product per CUDA thread.
    #[kernel]
    #[launch_bounds(256)]
    pub fn edge_products(
        column_indices: &[u32],
        weights: &[f32],
        input: &[f32],
        mut partials: DisjointSlice<f32>,
    ) {
        let edge = thread::index_1d();
        let edge_index = edge.get();
        let Some(destination) = partials.get_mut(edge) else {
            return;
        };
        let Some(column) = column_indices.get(edge_index).copied() else {
            return;
        };
        let Some(weight) = weights.get(edge_index).copied() else {
            return;
        };
        let Some(column_index) = usize::try_from(column).ok() else {
            return;
        };
        let Some(value) = input.get(column_index).copied() else {
            return;
        };
        *destination = weight * value;
    }

    /// Reduces the edge products in CSR row order into one output per row.
    #[kernel]
    #[launch_bounds(256)]
    pub fn reduce_rows(row_offsets: &[u32], partials: &[f32], mut output: DisjointSlice<f32>) {
        let row = thread::index_1d();
        let row_index = row.get();
        let Some(destination) = output.get_mut(row) else {
            return;
        };
        let Some(start) = row_offsets.get(row_index).copied() else {
            return;
        };
        let Some(next_row) = row_index.checked_add(1) else {
            return;
        };
        let Some(end) = row_offsets.get(next_row).copied() else {
            return;
        };
        let mut edge = start;
        let mut accumulator = 0.0_f32;
        while edge < end {
            let Some(edge_index) = usize::try_from(edge).ok() else {
                return;
            };
            let Some(value) = partials.get(edge_index).copied() else {
                return;
            };
            accumulator += value;
            let Some(next_edge) = edge.checked_add(1) else {
                return;
            };
            edge = next_edge;
        }
        *destination = accumulator;
    }
}

/// Creates the CUDA context and loads the generated kernel module.
pub(crate) fn new_cuda(device_ordinal: usize) -> Result<CudaSpmv, BenchError> {
    let context = CudaContext::new(device_ordinal)?;
    let ptx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../zkfly_bench.ptx");
    let ptx = fs::read_to_string(&ptx_path).map_err(|source| BenchError::PtxIo {
        path: ptx_path,
        source,
    })?;
    let module = context.load_module_from_ptx_src(&ptx)?;
    let module = kernels::from_module(module)?;
    Ok(CudaSpmv {
        context,
        module,
        execution_lock: Mutex::new(()),
    })
}

/// Enqueues the row-per-thread CSR kernel on one stream.
fn launch_row_kernel(
    runner: &CudaSpmv,
    stream: &CudaStream,
    launch: LaunchConfig,
    row_offsets: &DeviceBuffer<u32>,
    column_indices: &DeviceBuffer<u32>,
    weights: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), BenchError> {
    runner.module.csr_spmv(
        stream,
        launch,
        row_offsets,
        column_indices,
        weights,
        input,
        output,
    )?;
    Ok(())
}

/// Enqueues edge products followed by CSR row-order reduction.
fn launch_edge_kernel(
    runner: &CudaSpmv,
    stream: &CudaStream,
    row_launch: LaunchConfig,
    edge_launch: LaunchConfig,
    row_offsets: &DeviceBuffer<u32>,
    column_indices: &DeviceBuffer<u32>,
    weights: &DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    partials: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<(), BenchError> {
    runner.module.edge_products(
        stream,
        edge_launch,
        column_indices,
        weights,
        input,
        partials,
    )?;
    runner
        .module
        .reduce_rows(stream, row_launch, row_offsets, partials, output)?;
    Ok(())
}

/// Runs the row-per-thread kernel and compares the final output with CPU.
pub(crate) fn run_cuda(
    artifact: &CsrArtifact,
    input: &[f32],
    cpu_output: &[f32],
    rows: usize,
    config: BenchmarkConfig,
) -> Result<CudaMeasurement, BenchError> {
    if config.device_resident {
        run_cuda_resident(artifact, input, cpu_output, rows, config)
    } else {
        run_cuda_reallocating(artifact, input, cpu_output, rows, config)
    }
}

/// Runs the original allocation-per-iteration baseline.
fn run_cuda_reallocating(
    artifact: &CsrArtifact,
    input: &[f32],
    cpu_output: &[f32],
    rows: usize,
    config: BenchmarkConfig,
) -> Result<CudaMeasurement, BenchError> {
    let runner = new_cuda(config.device_ordinal)?;
    let guard = runner
        .execution_lock
        .lock()
        .map_err(|_| BenchError::RuntimeStatePoisoned)?;
    runner.context.bind_to_thread()?;
    let stream = runner.context.default_stream();
    let row_count = u32::try_from(rows).map_err(|_| BenchError::Configuration {
        proposition: "rows fit CUDA launch index type",
    })?;
    let grid_size = row_count.div_ceil(BLOCK_SIZE);
    let launch = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut upload = Duration::ZERO;
    let mut compute_download = Duration::ZERO;
    let mut end_to_end = Duration::ZERO;
    let mut final_output = Vec::new();
    for iteration in 0..config.warmup.saturating_add(config.iterations) {
        let end_start = Instant::now();
        let upload_start = Instant::now();
        let row_offsets = DeviceBuffer::from_host(&stream, &artifact.row_offsets)?;
        let column_indices = DeviceBuffer::from_host(&stream, &artifact.column_indices)?;
        let weights = DeviceBuffer::from_host(&stream, &artifact.weights)?;
        let input_device = DeviceBuffer::from_host(&stream, input)?;
        let mut output_device = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
        let upload_elapsed = upload_start.elapsed();
        let compute_start = Instant::now();
        runner.module.csr_spmv(
            &stream,
            launch,
            &row_offsets,
            &column_indices,
            &weights,
            &input_device,
            &mut output_device,
        )?;
        let output = output_device.to_host_vec(&stream)?;
        if iteration >= config.warmup {
            upload = upload.saturating_add(upload_elapsed);
            compute_download = compute_download.saturating_add(compute_start.elapsed());
            end_to_end = end_to_end.saturating_add(end_start.elapsed());
            final_output = output;
        }
    }
    drop(guard);
    let max_abs_error = max_abs_error(&final_output, cpu_output)?;
    Ok(CudaMeasurement {
        upload,
        compute_download,
        end_to_end,
        kernel: None,
        download: None,
        max_abs_error,
        device_resident: false,
    })
}

/// Runs the kernel with topology and vectors resident on the device.
fn run_cuda_resident(
    artifact: &CsrArtifact,
    input: &[f32],
    cpu_output: &[f32],
    rows: usize,
    config: BenchmarkConfig,
) -> Result<CudaMeasurement, BenchError> {
    let runner = new_cuda(config.device_ordinal)?;
    let guard = runner
        .execution_lock
        .lock()
        .map_err(|_| BenchError::RuntimeStatePoisoned)?;
    runner.context.bind_to_thread()?;
    let stream = runner.context.default_stream();
    let launch = launch_config(rows)?;
    let edge_launch = if config.edge_parallel {
        Some(launch_config(artifact.column_indices.len())?)
    } else {
        None
    };
    let upload_start = Instant::now();
    let row_offsets = DeviceBuffer::from_host(&stream, &artifact.row_offsets)?;
    let column_indices = DeviceBuffer::from_host(&stream, &artifact.column_indices)?;
    let weights = DeviceBuffer::from_host(&stream, &artifact.weights)?;
    let input_device = DeviceBuffer::from_host(&stream, input)?;
    let mut output_device = DeviceBuffer::<f32>::zeroed(&stream, rows)?;
    let mut partials = if config.edge_parallel {
        Some(DeviceBuffer::<f32>::zeroed(
            &stream,
            artifact.column_indices.len(),
        )?)
    } else {
        None
    };
    stream.synchronize()?;
    let upload = upload_start.elapsed();
    for _ in 0..config.warmup {
        if let (Some(partials), Some(edge_launch)) = (partials.as_mut(), edge_launch) {
            launch_edge_kernel(
                &runner,
                &stream,
                launch,
                edge_launch,
                &row_offsets,
                &column_indices,
                &weights,
                &input_device,
                partials,
                &mut output_device,
            )?;
        } else {
            launch_row_kernel(
                &runner,
                &stream,
                launch,
                &row_offsets,
                &column_indices,
                &weights,
                &input_device,
                &mut output_device,
            )?;
        }
    }
    stream.synchronize()?;
    let start_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let end_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let mut kernel = Duration::ZERO;
    let mut download = Duration::ZERO;
    let mut end_to_end = Duration::ZERO;
    let mut final_output = Vec::new();
    for _ in 0..config.iterations {
        let end_start = Instant::now();
        start_event.record(&stream)?;
        if let (Some(partials), Some(edge_launch)) = (partials.as_mut(), edge_launch) {
            launch_edge_kernel(
                &runner,
                &stream,
                launch,
                edge_launch,
                &row_offsets,
                &column_indices,
                &weights,
                &input_device,
                partials,
                &mut output_device,
            )?;
        } else {
            launch_row_kernel(
                &runner,
                &stream,
                launch,
                &row_offsets,
                &column_indices,
                &weights,
                &input_device,
                &mut output_device,
            )?;
        }
        end_event.record(&stream)?;
        let kernel_ms = start_event.elapsed_ms(&end_event)?;
        let download_start = Instant::now();
        final_output = output_device.to_host_vec(&stream)?;
        download = download.saturating_add(download_start.elapsed());
        kernel = kernel.saturating_add(event_duration(kernel_ms)?);
        end_to_end = end_to_end.saturating_add(end_start.elapsed());
    }
    drop(guard);
    let max_abs_error = max_abs_error(&final_output, cpu_output)?;
    Ok(CudaMeasurement {
        upload,
        compute_download: kernel.saturating_add(download),
        end_to_end,
        kernel: Some(kernel),
        download: Some(download),
        max_abs_error,
        device_resident: true,
    })
}

/// Builds the one-dimensional launch configuration used by both modes.
fn launch_config(rows: usize) -> Result<LaunchConfig, BenchError> {
    let row_count = u32::try_from(rows).map_err(|_| BenchError::Configuration {
        proposition: "rows fit CUDA launch index type",
    })?;
    let grid_size = row_count.div_ceil(BLOCK_SIZE);
    Ok(LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Converts a finite CUDA event duration into the host duration type.
fn event_duration(milliseconds: f32) -> Result<Duration, BenchError> {
    if !milliseconds.is_finite() || milliseconds < 0.0 {
        return Err(BenchError::Configuration {
            proposition: "CUDA event duration is finite and nonnegative",
        });
    }
    Ok(Duration::from_secs_f64(f64::from(milliseconds) / 1_000.0))
}

/// Finds the largest elementwise absolute error between two vectors.
fn max_abs_error(actual: &[f32], expected: &[f32]) -> Result<f32, BenchError> {
    if actual.len() != expected.len() {
        return Err(BenchError::Configuration {
            proposition: "CUDA and CPU output lengths match",
        });
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (*left - *right).abs())
        .fold(0.0_f32, f32::max))
}
